#!/usr/bin/env python3
"""Offline unit test for sim2real_bridge safety gates (no unitree_sdk2py/DDS needed).

Stubs unitree_sdk2py + redis modules, then verifies each of the 7 safety gates
behaves correctly:

  G1: --enable required (no publish when disabled)
  G2: mode_machine must match expected (no publish on mismatch)
  G3: joint/inspire range + finite checks (refuse out-of-range / NaN)
  G4: motor_cmd.mode only set to 1 after all checks pass (never when refused)
  G5: e-stop key `sim2real_estop`==1 stops publishing
  G6: stale ts (> timeout_s) refuses publish; TTL set on cmd key
  G7: --topic-prefix redirects publish topics to rt/<prefix>/...

Run: python tools/test_sim2real_bridge_safety.py
"""
from __future__ import annotations

import sys
import types
import time
import json
import importlib.util
from pathlib import Path

BRIDGE_PATH = Path(__file__).resolve().parent / "sim2real_bridge.py"


class _FakeCmd:
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class _FakeLowCmd:
    def __init__(self):
        self.mode_machine = 0
        self.mode_pr = 0
        self.motor_cmd = [_FakeCmd() for _ in range(35)]
        self.crc = 0


class _FakeMotorCmds:
    def __init__(self):
        self.cmds = [_FakeCmd() for _ in range(12)]


class _FakeLowState:
    def __init__(self, mode_machine=0):
        self.mode_machine = mode_machine


def _install_fakes():
    redis_mod = types.ModuleType("redis")
    redis_lib = types.ModuleType("redis_lib")

    class FakeRedis:
        def __init__(self, host, port, decode_responses=False):
            self.data = {}
            self.decoded = decode_responses

        def get(self, k):
            return self.data.get(k)

        def set(self, k, v):
            self.data[k] = v

        def delete(self, *ks):
            for k in ks:
                self.data.pop(k, None)

        def expire(self, k, s):
            self.data[k + ":ttl"] = s

    redis_lib.Redis = FakeRedis
    redis_mod.Redis = FakeRedis
    sys.modules["redis"] = redis_mod

    # unitree_sdk2py stubs
    sdk = types.ModuleType("unitree_sdk2py")
    core = types.ModuleType("unitree_sdk2py.core")
    channel = types.ModuleType("unitree_sdk2py.core.channel")
    idl = types.ModuleType("unitree_sdk2py.idl")
    default = types.ModuleType("unitree_sdk2py.idl.default")
    go = types.ModuleType("unitree_sdk2py.idl.unitree_go")
    hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg")
    gomsg = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg")
    hgmsg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg")
    dds_go = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
    dds_hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    utils = types.ModuleType("unitree_sdk2py.utils")
    crc_mod = types.ModuleType("unitree_sdk2py.utils.crc")

    class FakeCRC:
        def Crc(self, obj):
            return 12345

    crc_mod.CRC = FakeCRC

    dds_go.MotorCmds_ = _FakeMotorCmds
    dds_go.MotorStates_ = object
    dds_hg.LowCmd_ = _FakeLowCmd
    dds_hg.LowState_ = _FakeLowState

    default.unitree_go_msg_dds__MotorCmd_ = _FakeCmd
    default.unitree_hg_msg_dds__LowCmd_ = _FakeLowCmd

    class FakeChannel:
        def __init__(self, topic, cls):
            self.topic = topic
            self.cls = cls
            self.writes = []

        def Init(self, *a, **k):
            pass

        def Write(self, msg):
            self.writes.append(msg)

    class FakePub(FakeChannel):
        pass

    class FakeSub(FakeChannel):
        def Init(self, cb, q):
            self.cb = cb
            self.q = q

    channel.ChannelFactoryInitialize = lambda domain: None
    channel.ChannelPublisher = FakePub
    channel.ChannelSubscriber = FakeSub

    sys.modules["unitree_sdk2py"] = sdk
    sys.modules["unitree_sdk2py.core"] = core
    sys.modules["unitree_sdk2py.core.channel"] = channel
    sys.modules["unitree_sdk2py.idl"] = idl
    sys.modules["unitree_sdk2py.idl.default"] = default
    sys.modules["unitree_sdk2py.idl.unitree_go"] = go
    sys.modules["unitree_sdk2py.idl.unitree_hg"] = hg
    sys.modules["unitree_sdk2py.idl.unitree_go.msg"] = gomsg
    sys.modules["unitree_sdk2py.idl.unitree_hg.msg"] = hgmsg
    sys.modules["unitree_sdk2py.idl.unitree_go.msg.dds_"] = dds_go
    sys.modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"] = dds_hg
    sys.modules["unitree_sdk2py.utils"] = utils
    sys.modules["unitree_sdk2py.utils.crc"] = crc_mod


def _load_bridge():
    spec = importlib.util.spec_from_file_location("s2r_bridge", BRIDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _payload(body=None, ts=None, inspire_r=None, inspire_l=None):
    p = {}
    if body is not None:
        p["body_29"] = list(body)
    if ts is not None:
        p["ts"] = int(ts * 1e9)
    if inspire_r is not None:
        p["inspire_right_12"] = list(inspire_r)
    if inspire_l is not None:
        p["inspire_left_12"] = list(inspire_l)
    return p


def _make_bridge(mod, enable=False, prefix="", mode=0, **kw):
    b = mod.Sim2RealBridge(
        "localhost", 6379, 50.0, domain=0,
        enable=enable, topic_prefix=prefix,
        expected_mode_machine=mode, **kw,
    )
    # access the fake publishers via attributes
    return b


def main():
    _install_fakes()
    mod = _load_bridge()

    results = []

    def check(name, cond):
        results.append((name, cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    body_ok = [0.0] * 29
    body_ok[1] = 0.5
    body_ok[28] = -1.2
    insp12 = [0.5] * 12
    now = time.time()

    # G1: disabled bridge never publishes
    b = _make_bridge(mod, enable=False)
    b._on_lowstate(_FakeLowState(0))
    b._maybe_publish(_payload(body_ok, now, insp12, insp12))
    check("G1 disabled -> no lowcmd write", len(b.lowcmd_pub.writes) == 0)
    check("G1 disabled -> no inspire write", len(b.inspire_pub.writes) == 0)

    # G1: enabled bridge publishes when everything is fine
    b = _make_bridge(mod, enable=True)
    b._on_lowstate(_FakeLowState(0))
    b._maybe_publish(_payload(body_ok, now, insp12, insp12))
    check("G1 enabled+valid -> lowcmd written", len(b.lowcmd_pub.writes) == 1)
    check("G1 enabled+valid -> inspire written", len(b.inspire_pub.writes) == 1)
    check("G4 motor mode==1 after pass",
          all(m.mode == 1 for m in b.low_cmd.motor_cmd[:29]))

    # G2: mode_machine mismatch
    b = _make_bridge(mod, enable=True, mode=0)
    b._on_lowstate(_FakeLowState(5))
    b._maybe_publish(_payload(body_ok, now, insp12, insp12))
    check("G2 mode_machine mismatch -> no publish", len(b.lowcmd_pub.writes) == 0)

    # G2: no lowstate yet
    b = _make_bridge(mod, enable=True)
    b._maybe_publish(_payload(body_ok, now, insp12, insp12))
    check("G2 no lowstate -> no publish", len(b.lowcmd_pub.writes) == 0)

    # G3: out-of-range body
    bad = list(body_ok)
    bad[0] = 2.5
    b = _make_bridge(mod, enable=True)
    b._on_lowstate(_FakeLowState(0))
    b._maybe_publish(_payload(bad, now, insp12, insp12))
    check("G3 body out of range -> no publish", len(b.lowcmd_pub.writes) == 0)
    check("G4 motor mode stays 0 on refuse",
          all(m.mode == 0 for m in b.low_cmd.motor_cmd))

    # G3: NaN
    b = _make_bridge(mod, enable=True)
    b._on_lowstate(_FakeLowState(0))
    nan = list(body_ok)
    nan[3] = float("nan")
    b._maybe_publish(_payload(nan, now, insp12, insp12))
    check("G3 NaN body -> no publish", len(b.lowcmd_pub.writes) == 0)

    # G3: inspire out of range
    b = _make_bridge(mod, enable=True)
    b._on_lowstate(_FakeLowState(0))
    b._maybe_publish(_payload(body_ok, now, [3.9] + [0.5] * 11, insp12))
    check("G3 inspire out of range -> no publish", len(b.inspire_pub.writes) == 0)

    # G5: e-stop
    b = _make_bridge(mod, enable=True)
    b._on_lowstate(_FakeLowState(0))
    b._redis.set(mod.ESTOP_KEY, "1")
    b._maybe_publish(_payload(body_ok, now, insp12, insp12))
    check("G5 estop=1 -> no publish", len(b.lowcmd_pub.writes) == 0)
    b._redis.delete(mod.ESTOP_KEY)
    b._maybe_publish(_payload(body_ok, now, insp12, insp12))
    check("G5 estop cleared -> publish resumes", len(b.lowcmd_pub.writes) == 1)

    # G6: stale command
    b = _make_bridge(mod, enable=True, timeout_s=0.5)
    b._on_lowstate(_FakeLowState(0))
    b._maybe_publish(_payload(body_ok, now - 5.0, insp12, insp12))
    check("G6 stale ts -> no publish", len(b.lowcmd_pub.writes) == 0)

    # G7: topic prefix redirect
    b = _make_bridge(mod, enable=True, prefix="safe_test")
    check("G7 lowcmd topic prefixed",
          b.lowcmd_pub.topic == "rt/safe_test/lowcmd")
    check("G7 inspire topic prefixed",
          b.inspire_pub.topic == "rt/safe_test/inspire/cmd")
    check("G7 lowstate topic prefixed",
          b.lowstate_sub.topic == "rt/safe_test/lowstate")

    # G7: default (no prefix) = production topics
    b = _make_bridge(mod, enable=True)
    check("G7 default lowcmd topic production",
          b.lowcmd_pub.topic == "rt/lowcmd")
    check("G7 default inspire topic production",
          b.inspire_pub.topic == "rt/inspire/cmd")

    # G4: CRC computed after fill
    b = _make_bridge(mod, enable=True)
    b._on_lowstate(_FakeLowState(0))
    b._maybe_publish(_payload(body_ok, now, insp12, insp12))
    check("G4 CRC set on published lowcmd", b.low_cmd.crc == 12345)

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} gates passed")
    if failed:
        print("FAILED:", failed)
        return 1
    print("ALL SAFETY GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
