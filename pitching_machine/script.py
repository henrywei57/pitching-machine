"""PitchScript: a little language for controlling the pitching machine. Files end in .pitch.

    pitchscript my_pitches.pitch
    python -m pitching_machine.script my_pitches.pitch      (same thing)

A .pitch file looks like this:

    x = pitch(3200, 2600, 2000)        # wheel rpm: top, lower 3B, lower 1B
    print(x.result())
    x.show()

    a = pitch(3200, 2600, 2000, color="red")
    b = pitch("knuckleball", color="blue")
    show(a, b)                         # both at once, each with its own color and trail

The syntax is borrowed from Python (same comments, math, if/for/while, f-strings),
but this file runs it with its own small interpreter: only the things listed in
LANGUAGE below exist, and nothing else from Python does (no import, no def,
no files).
"""

import ast
import operator
import os
import sys
from pathlib import Path

import numpy as np

from .constants import FOOT, INCH, MPH, RPM
from .machine import calibrate, settings_for
from .pitch import PRESETS, plot_pitch, preset_config, rpm_config, throw
from .show import pick_colors, play, rgb, save_gif

LANGUAGE = """\
Making pitches
  pitch(3200, 2600, 2000)              wheel rpm: top, lower 3B, lower 1B
  pitch("curveball")                   a preset: fastball, curveball, knuckle-curve, knuckleball
  pitch(speed=85, spin=-2000, axis=15) the pitch you want; the wheels are worked out for you
      add calibrate=True               ...and keep adjusting the wheels until it really comes out that way
  extra options                        color="red", seed=3 (how the ball sits in the chute), friction=0.9

What a pitch can tell you
  x.result()       the full report, as text (print it)
  x.speed          exit speed, mph              x.spin     exit spin, rpm (+ topspin)
  x.axis           spin axis, deg               x.wheels   the three wheel rpms
  x.plate_speed    speed at the plate, mph      x.time     flight time, s
  x.height         height at the plate, ft      x.side     inches toward 1B (- is toward 3B)
  x.hbreak         sideways break, in           x.vbreak   up/down break, in
  x.color          its ball color

Watching
  x.show()                     open the 3D viewer with this pitch
  show(a, b, c)                several pitches at once (a list works too: show(my_list))
  show(a, b, camera="catcher") cameras: watcher, catcher, batter, side, machine, follow_ball
  gif("file.gif", a, b)        save the animation as a GIF instead of opening a window
  x.plot("file.png")           save a side view and top view of the flight

Everything else
  print(...)  range(...)  len(...)  round(...)  abs(...)  min(...)  max(...)  sum(...)
  int(...)  float(...)  str(...)  list.append(...)
  numbers, "text", f"text {x.speed:.1f}", [lists], + - * / // % **, == != < <= > >=, and or not
  if / elif / else,  for v in ...,  while ...,  break, continue
  # comments
"""

MAX_LOOP = 100_000


class ScriptError(Exception):
    """A mistake in a .pitch file, with the line it happened on."""

    def __init__(self, message, line=None):
        super().__init__(message)
        self.line = line


class ScriptPitch:
    """What a .pitch file gets back from pitch(): read-only numbers and a few actions."""

    ATTRS = {"result", "show", "gif", "plot", "speed", "spin", "axis", "wheels", "plate_speed", "time",
             "height", "side", "hbreak", "vbreak", "color"}

    def __init__(self, p, color, label):
        self._p, self.color, self._label = p, color, label

    @property
    def speed(self):
        return round(self._p.launch.speed / MPH, 2)

    @property
    def spin(self):
        return round(self._p.launch.signed_spin / RPM, 1)

    @property
    def axis(self):
        return round(self._p.launch.spin_axis_deg, 1)

    @property
    def wheels(self):
        return [round(r, 1) for r in self._p.launch.config.wheel_rpm]

    def _plate(self, value):
        return round(value, 2) if np.isfinite(self._p.flight_time) else None

    @property
    def plate_speed(self):
        return self._plate(self._p.plate_speed / MPH)

    @property
    def time(self):
        return self._plate(self._p.flight_time)

    @property
    def height(self):
        return self._plate(self._p.plate_height / FOOT)

    @property
    def side(self):
        return self._plate(self._p.plate_side / INCH)

    @property
    def hbreak(self):
        return self._plate(self._p.horizontal_break)

    @property
    def vbreak(self):
        return self._plate(self._p.vertical_break)

    def result(self):
        return self._p.summary()

    def show(self, camera="watcher"):
        _show([self], camera)

    def gif(self, path, camera="watcher"):
        return _gif(path, [self], camera)

    def plot(self, path):
        plot_pitch(self._p, path)
        return path

    def __repr__(self):
        where = f", {self.height:.2f} ft high" if self.height is not None else ", hit the ground"
        return f"{self._label} -> {self.speed:.1f} mph, {self.spin:+.0f} rpm{where} ({self.color})"


# ----------------------------------------------------------------------------------
# Built-in functions
# ----------------------------------------------------------------------------------

_used_colors = []


def make_pitch(*args, color=None, seed=None, friction=None, speed=None, spin=None, axis=0.0, calibrate=False):
    over = {} if friction is None else {"friction": float(friction)}
    if speed is not None or spin is not None:
        if args:
            raise ScriptError("give either wheel rpms or speed=/spin=, not both")
        if speed is None or spin is None:
            raise ScriptError("pitch(speed=..., spin=...) needs both a speed (mph) and a spin (rpm)")
        if calibrate:
            res = _calibrate(float(speed), float(spin), float(axis), **over)
            cfg = res.config
        else:
            cfg = settings_for(float(speed), float(spin), float(axis), **over)
        label = f"pitch(speed={speed}, spin={spin}, axis={axis})"
    elif len(args) == 1 and isinstance(args[0], str):
        name = args[0].lower()
        if name not in PRESETS:
            raise ScriptError(f"there's no preset called {args[0]!r}; try {', '.join(PRESETS)}")
        cfg = preset_config(name, **over)
        label = f'pitch("{name}")'
    elif len(args) == 3 and all(isinstance(a, (int, float)) for a in args):
        cfg = rpm_config([float(a) for a in args], **over)
        label = "pitch(" + ", ".join(f"{a:g}" for a in args) + ")"
    else:
        raise ScriptError("pitch() needs 3 wheel rpms (top, lower 3B, lower 1B), a preset name like "
                          "\"curveball\", or speed= and spin=")
    cfg.check_rpm()
    if seed is not None:
        q = np.random.default_rng(int(seed)).normal(size=4)
        cfg = cfg.with_(ball_quat=tuple(q / np.linalg.norm(q)))
    if color is not None:
        rgb(color)  # check it now, so the error points at this line
    try:
        p = throw(cfg)
    except RuntimeError:
        raise ScriptError(f"with {label} the ball never leaves the wheels; turn them up") from None
    if color is None:
        color = pick_colors([*_used_colors, None])[-1]
    _used_colors.append(color)
    return ScriptPitch(p, color, label)


def _calibrate(speed, spin, axis, **kw):
    res = calibrate(speed, spin, axis, **kw)
    if res is None:
        raise ScriptError("the machine can't get anywhere near that pitch")
    return res


def _flatten(items):
    out = []
    for it in items:
        if isinstance(it, (list, tuple)):
            out += _flatten(it)
        elif isinstance(it, ScriptPitch):
            out.append(it)
        else:
            raise ScriptError(f"show() and gif() take pitches, not {_kind(it)}")
    if not out:
        raise ScriptError("give show() or gif() at least one pitch")
    return out


def _show(pitches, camera="watcher"):
    from .view import CAMERAS

    if camera not in CAMERAS:
        raise ScriptError(f"there's no {camera!r} camera; try {', '.join(CAMERAS)}")
    import mujoco.viewer

    if sys.platform == "darwin" and mujoco.viewer._MJPYTHON is None:
        raise ScriptError("on a Mac, show() needs mjpython: run this file with pitchscript or mjpython")
    print("showing " + ", ".join(p.color for p in pitches) + " (close the window to go on)")
    play([p._p for p in pitches], [p.color for p in pitches], camera=camera)


def builtin_show(*items, camera="watcher"):
    _show(_flatten(items), camera)


def _gif(path, pitches, camera="watcher"):
    path = str(path)
    if not path.lower().endswith(".gif"):
        raise ScriptError("gif() saves a .gif file, like gif(\"pitches.gif\", a, b)")
    save_gif(path, [p._p for p in pitches], [p.color for p in pitches], camera=camera)
    print(f"saved {path}")
    return path


def builtin_gif(path, *items, camera="watcher"):
    return _gif(path, _flatten(items), camera)


def _kind(v):
    return {int: "a number", float: "a number", str: "text", bool: "True/False", list: "a list",
            type(None): "nothing (None)", ScriptPitch: "a pitch"}.get(type(v), type(v).__name__)


def _print(*args, sep=" ", end="\n"):
    print(*args, sep=sep, end=end)


def _range(*args):
    if not 1 <= len(args) <= 3 or not all(isinstance(a, int) for a in args):
        raise ScriptError("range() takes 1 to 3 whole numbers, like range(2000, 4001, 500)")
    r = range(*args)
    if len(r) > MAX_LOOP:
        raise ScriptError(f"that range has {len(r)} numbers; the limit is {MAX_LOOP}")
    return list(r)


BUILTINS = {
    "pitch": make_pitch, "show": builtin_show, "gif": builtin_gif, "print": _print, "range": _range,
    "len": len, "round": round, "abs": abs, "min": min, "max": max, "sum": sum,
    "int": int, "float": float, "str": str,
    "True": True, "False": False, "None": None,
}

BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
CMPOPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
          ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
          ast.NotIn: lambda a, b: a not in b}
UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
# Methods a script may call on plain values.
VALUE_METHODS = {list: {"append"}, str: {"upper", "lower"}}

FRIENDLY = {
    ast.FunctionDef: "making your own functions (def)", ast.Import: "import", ast.ImportFrom: "import",
    ast.ClassDef: "classes", ast.Lambda: "lambda", ast.With: "with", ast.Try: "try/except",
    ast.Return: "return", ast.Global: "global", ast.Delete: "del", ast.Dict: "dictionaries ({...})",
    ast.ListComp: "list comprehensions", ast.Raise: "raise", ast.AsyncFunctionDef: "async",
}


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class Interpreter:
    def __init__(self):
        self.vars = {}

    # -- running ------------------------------------------------------------------

    def run(self, source, filename="<script>"):
        _used_colors.clear()
        try:
            tree = ast.parse(source, filename)
        except SyntaxError as e:
            raise ScriptError(f"I can't read this line: {e.msg}", e.lineno) from None
        self.block(tree.body)
        return self.vars

    def block(self, stmts):
        for s in stmts:
            try:
                self.statement(s)
            except (ScriptError, _Break, _Continue) as e:
                if isinstance(e, ScriptError) and e.line is None:
                    e.line = s.lineno
                raise
            except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                raise ScriptError(_explain(e), s.lineno) from None

    # -- statements ---------------------------------------------------------------

    def statement(self, s):
        if isinstance(s, ast.Expr):
            self.expr(s.value)
        elif isinstance(s, ast.Assign):
            value = self.expr(s.value)
            for target in s.targets:
                self.assign(target, value)
        elif isinstance(s, ast.AugAssign):
            if not isinstance(s.target, ast.Name):
                raise ScriptError("+= and friends only work on a plain name, like total += 1")
            op = BINOPS.get(type(s.op))
            if op is None:
                raise ScriptError("that kind of math isn't in PitchScript")
            self.vars[s.target.id] = op(self.name(s.target), self.expr(s.value))
        elif isinstance(s, ast.If):
            self.block(s.body if self.truth(s.test) else s.orelse)
        elif isinstance(s, ast.For):
            items = self.expr(s.iter)
            if not isinstance(items, (list, tuple, str)):
                raise ScriptError(f"for ... in needs a list or range(), not {_kind(items)}")
            for item in items:
                self.assign(s.target, item)
                try:
                    self.block(s.body)
                except _Break:
                    break
                except _Continue:
                    continue
            else:
                self.block(s.orelse)
        elif isinstance(s, ast.While):
            n = 0
            while self.truth(s.test):
                n += 1
                if n > MAX_LOOP:
                    raise ScriptError(f"this while loop ran {MAX_LOOP} times; is it stuck?")
                try:
                    self.block(s.body)
                except _Break:
                    break
                except _Continue:
                    continue
        elif isinstance(s, ast.Break):
            raise _Break()
        elif isinstance(s, ast.Continue):
            raise _Continue()
        elif isinstance(s, ast.Pass):
            pass
        else:
            raise ScriptError(f"PitchScript doesn't have {FRIENDLY.get(type(s), type(s).__name__)}")

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            if target.id in BUILTINS:
                raise ScriptError(f"'{target.id}' is a built-in name; pick another variable name")
            self.vars[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            value = list(value)
            if len(value) != len(target.elts):
                raise ScriptError(f"there are {len(target.elts)} names on the left but {len(value)} values")
            for t, v in zip(target.elts, value):
                self.assign(t, v)
        elif isinstance(target, ast.Subscript):
            seq = self.expr(target.value)
            if not isinstance(seq, list):
                raise ScriptError("you can only change items of a list")
            seq[self.expr(target.slice)] = value
        else:
            raise ScriptError("you can only assign to a name, like x = ...")

    # -- expressions --------------------------------------------------------------

    def truth(self, node):
        return bool(self.expr(node))

    def name(self, node):
        if node.id in self.vars:
            return self.vars[node.id]
        if node.id in BUILTINS:
            return BUILTINS[node.id]
        raise ScriptError(f"I don't know what '{node.id}' is (did you set it first?)")

    def expr(self, n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return self.name(n)
        if isinstance(n, ast.BinOp):
            op = BINOPS.get(type(n.op))
            if op is None:
                raise ScriptError("that kind of math isn't in PitchScript")
            return op(self.expr(n.left), self.expr(n.right))
        if isinstance(n, ast.UnaryOp):
            return UNARY[type(n.op)](self.expr(n.operand))
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                v = True
                for e in n.values:
                    v = self.expr(e)
                    if not v:
                        return v
                return v
            for e in n.values:
                v = self.expr(e)
                if v:
                    return v
            return v
        if isinstance(n, ast.Compare):
            left = self.expr(n.left)
            for op, right_node in zip(n.ops, n.comparators):
                right = self.expr(right_node)
                if type(op) not in CMPOPS:
                    raise ScriptError("use == or != to compare")
                if not CMPOPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return self.expr(n.body) if self.truth(n.test) else self.expr(n.orelse)
        if isinstance(n, (ast.List, ast.Tuple)):
            return [self.expr(e) for e in n.elts]
        if isinstance(n, ast.Subscript):
            return self.expr(n.value)[self.expr(n.slice)]
        if isinstance(n, ast.Slice):
            return slice(*(None if x is None else self.expr(x) for x in (n.lower, n.upper, n.step)))
        if isinstance(n, ast.JoinedStr):
            return "".join(self.expr(v) if isinstance(v, ast.Constant) else self.formatted(v) for v in n.values)
        if isinstance(n, ast.Attribute):
            return self.attribute(self.expr(n.value), n.attr)
        if isinstance(n, ast.Call):
            return self.call(n)
        raise ScriptError(f"PitchScript doesn't have {FRIENDLY.get(type(n), type(n).__name__)}")

    def formatted(self, v):
        value = self.expr(v.value)
        spec = self.expr(v.format_spec) if v.format_spec is not None else ""
        if v.conversion == ord("r"):
            value = repr(value)
        return format(value, spec)

    def attribute(self, obj, attr):
        if isinstance(obj, ScriptPitch):
            if attr not in ScriptPitch.ATTRS:
                raise ScriptError(f"a pitch doesn't have '.{attr}'; it has {', '.join(sorted(ScriptPitch.ATTRS))}")
            return getattr(obj, attr)
        allowed = VALUE_METHODS.get(type(obj), set())
        if attr not in allowed:
            raise ScriptError(f"{_kind(obj)} doesn't have '.{attr}'")
        return getattr(obj, attr)

    def call(self, n):
        fn = self.expr(n.func)
        if not callable(fn):
            raise ScriptError(f"{_kind(fn)} can't be called like a function")
        args = []
        for a in n.args:
            if isinstance(a, ast.Starred):
                args += list(self.expr(a.value))
            else:
                args.append(self.expr(a))
        kwargs = {}
        for k in n.keywords:
            if k.arg is None:
                raise ScriptError("** isn't in PitchScript")
            kwargs[k.arg] = self.expr(k.value)
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            raise ScriptError(_explain(e)) from None


def _explain(e):
    msg = str(e)
    if isinstance(e, ZeroDivisionError):
        return "dividing by zero"
    if "unexpected keyword argument" in msg:
        bad = msg.split("'")[1] if "'" in msg else "?"
        return f"there's no option called {bad}="
    return msg


# ----------------------------------------------------------------------------------
# Running files
# ----------------------------------------------------------------------------------


def uses_viewer(source):
    """Does the file call show()? (It then has to run under mjpython on a Mac.)"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any((isinstance(n, ast.Name) and n.id == "show") or (isinstance(n, ast.Attribute) and n.attr == "show")
               for n in ast.walk(tree))


def run_file(path):
    path = Path(path)
    source = path.read_text()
    Interpreter().run(source, str(path))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.split("The syntax is borrowed")[0].rstrip())
        print("\nEverything PitchScript knows:\n\n" + LANGUAGE)
        return 0
    path = Path(argv[0])
    if not path.exists():
        print(f"can't find {path}")
        return 1
    if path.suffix != ".pitch":
        print(f"note: PitchScript files usually end in .pitch ({path.name})")
    source = path.read_text()

    # MuJoCo's viewer only runs under mjpython on macOS; restart under it if the file needs a window.
    if sys.platform == "darwin" and uses_viewer(source) and not os.environ.get("PITCHSCRIPT_UNDER_MJPYTHON"):
        import mujoco.viewer

        if mujoco.viewer._MJPYTHON is None:
            mj = Path(sys.executable).with_name("mjpython")
            if mj.exists():
                os.environ["PITCHSCRIPT_UNDER_MJPYTHON"] = "1"
                os.execv(str(mj), [str(mj), "-m", "pitching_machine.script", *argv])

    try:
        Interpreter().run(source, str(path))
    except ScriptError as e:
        where = f"{path.name}, line {e.line}" if e.line else path.name
        print(f"\n{where}: {e}")
        if e.line:
            print("    " + source.splitlines()[e.line - 1].strip())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
