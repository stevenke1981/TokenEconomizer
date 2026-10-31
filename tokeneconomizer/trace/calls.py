"""Every agent's tool calls as one ordered list of `(name, args, result_text, success)`.

`load_trajectory` reads a saved session log of any supported agent -- a Claude Code transcript
or `stream-json`, a Copilot CLI event log, a Codex `exec --json` stream -- and returns the same
tuples, so a mechanism never needs to know which agent wrote the file. A Codex rollout and a Pi
session are converted into one of these first (`codex.codex_rollout_to_exec`,
`pi.pi_session_to_claude`).
"""
import os
import re

from .claude import claude_calls
from .codex import codex_exec_calls
from .copilot import copilot_event_calls, copilot_session_cwd

# rtk in its Codex mode changes the COMMAND LINE rather than hooking it: the agent is told to
# type `rtk pytest`, `rtk git status`, `rtk proxy <cmd>`. Every classifier that reads a
# command's first token -- the micro-skill vocabulary, a test-run detector -- would otherwise
# stop seeing a test run, a fetch or a repository query. One definition, so the classifiers
# cannot disagree about what counts as the wrapper. `rtk proxy` is rtk's own passthrough (run
# unfiltered, still tracked); a bare `rtk` with no command is left alone.
RTK_PREFIX_RE = re.compile(r"^rtk\s+(?:proxy\s+)?(?=\S)")


def strip_rtk_prefix(command):
    """`rtk pytest -q` -> `pytest -q`; anything else unchanged."""
    return RTK_PREFIX_RE.sub("", (command or "").lstrip(), count=1)


def _is_codex_exec(lines):
    return any('"type":"item.completed"' in line.replace(" ", "") for line in lines)


def _read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [l.strip() for l in fh if l.strip()]


def _calls_and_cwds(lines):
    """Both `load_trajectory` and `call_cwds` over already-read lines, parsing them once."""
    if _is_codex_exec(lines):
        calls = codex_exec_calls(lines)
        return calls, [None] * len(calls)
    if any('"tool_use"' in line or '"type": "assistant"' in line or '"type":"assistant"' in line
           for line in lines):
        full = claude_calls(lines)
        return [c[:4] for c in full], [c[4] for c in full]
    calls = copilot_event_calls(lines)
    return calls, [copilot_session_cwd(lines)] * len(calls)


def load_trajectory_with_cwds(path):
    """`(load_trajectory(path), call_cwds(path))` from one read and one parse of the log, for a
    caller that needs both."""
    if not os.path.exists(path):
        return [], []
    return _calls_and_cwds(_read_lines(path))


def load_trajectory(path):
    """Ordered [(tool_name, args, result_text, success)] over a saved session log.

    The schema is sniffed from the lines: Codex emits completed items, Claude Code emits message
    envelopes carrying `tool_use` blocks, and anything else is read as Copilot events."""
    return load_trajectory_with_cwds(path)[0]


def call_cwds(path):
    """The directory each call of `load_trajectory(path)` ran in, aligned with it; None where
    the log does not say. Claude Code's shell keeps its directory between calls, and its
    transcript records it on every entry: after `cd tests && ...`, a bare `python runtests.py`
    runs in `tests/`. Codex runs each command afresh. Copilot's log records only where the
    session started, and that is where its calls are read as starting: models there type
    `cd <that directory> && ...` before every command, which is then the plain form."""
    return load_trajectory_with_cwds(path)[1]


def tool_usage(path):
    """{tool_name: calls} over a saved session log."""
    counts = {}
    for name, _args, _res, _ok in load_trajectory(path):
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts
