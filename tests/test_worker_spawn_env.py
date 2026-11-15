# SPDX-License-Identifier: MIT
"""Workers must not inherit ROCPROFILER_REGISTER_LIBRARY from their parent.

HIP init sets it with C setenv (so os.environ never sees it); a worker that
starts with it set records no GPU events in its kineto trace.
"""

import ctypes

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine import async_proc

VAR = b"ROCPROFILER_REGISTER_LIBRARY"
_libc = ctypes.CDLL(None)
_libc.getenv.restype = ctypes.c_char_p


class _Started(Exception):
    pass


class _Ctx:
    """Records the C environment a worker would inherit at its start()."""

    def __init__(self):
        self.env_at_start = []

    def Barrier(self, n):
        return None

    def Process(self, **kwargs):
        ctx = self

        class _Proc:
            def start(self):
                ctx.env_at_start.append(_libc.getenv(VAR))
                raise _Started

        return _Proc()


class _FakeMq:
    def export_handle(self):
        return None


def test_worker_does_not_inherit_rocprofiler_register_library(monkeypatch):
    ctx = _Ctx()
    monkeypatch.setattr(async_proc, "get_mp_context", lambda: ctx)
    monkeypatch.setattr(async_proc, "MessageQueue", lambda *a, **k: _FakeMq())
    monkeypatch.setattr(async_proc, "init_exit_handler", lambda mgr: None)
    monkeypatch.setattr("atexit.register", lambda fn: None)
    # As rocprofiler-register does: C setenv, absent from os.environ.
    assert _libc.setenv(VAR, b"/opt/rocm/lib/librocprofiler-sdk.so.1", 1) == 0
    try:
        with pytest.raises(_Started):
            async_proc.AsyncIOProcManager(lambda: None, 1, "x.Runner")
    finally:
        _libc.unsetenv(VAR)
    assert ctx.env_at_start == [None]
