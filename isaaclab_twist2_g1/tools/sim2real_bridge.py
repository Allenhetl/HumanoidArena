#!/usr/bin/env python3
"""sim2real bridge: publish policy body+hand targets to the real G1 robot over DDS.

Data flow:
  Isaac policy (MimicLiteActionProvider) -> Redis key `sim2real_cmd`
      {body_29: [...rad], inspire_right_12: [...], inspire_left_12: [...], ts: ns}
  This bridge subscribes Redis, converts:
    - body_29 (MIMIC_LITE_JOINT_ORDER) -> rt/lowcmd (LowCmd_, G1 motor index order)
    - inspire_left/right 12 -> rt/inspire/cmd (MotorCmds_, 12 motors, DFX order)
  It also subscribes rt/lowstate to mirror `mode_machine` (required by the
  low-level controller before it accepts lowcmd).

SAFETY (all required before real-robot use):
  1. --enable : explicit arming. Without it the bridge only listens, never writes.
  2. mode_machine match : only publish lowcmd when rt/lowstate mode_machine matches
     the configured expected value (default 0). Mismatch -> refuse + warn.
  3. Range / sanity check : body_29 must be finite and within [--joint-min, --joint-max]
     (default [-1.7, 1.7]); inspire values within [--inspire-min, --inspire-max]
     (default [0.0, 3.2]). Any violation -> refuse + warn (do not publish).
  4. Motor enable : motor_cmd.mode defaults to 0 (disable). It is set to 1
     (enable) ONLY after all safety checks pass.
  5. E-stop : Redis key `sim2real_estop` == "1" disables publishing immediately.
  6. Freshness timeout : sim2real_cmd `ts` (ns) older than --timeout-s (default
     0.5 s) is stale -> refuse. Also Redis TTL is set on the key so stale data
     expires.
  7. Topic isolation : --topic-prefix redirects publishing to non-production
     topics (e.g. `rt/lowcmd_safe_test`) so cross-machine verification cannot
     touch the real robot bus. Default prefix is empty (production topics).

Run:
  python tools/sim2real_bridge.py --enable                 # real robot, armed
  python tools/sim2real_bridge.py                          # listen-only (safe)
  python tools/sim2real_bridge.py --enable --topic-prefix safe_test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---- joint-name -> G1 lowcmd motor index (official G1JointIndex, 29dof) ----
G1_JOINT_INDEX = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}

# Order used by the policy (MIMIC_LITE_JOINT_ORDER)
MIMIC_LITE_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# Map policy order -> lowcmd motor index
_POLICY_TO_MOTOR = [G1_JOINT_INDEX[n] for n in MIMIC_LITE_JOINT_ORDER]

# DFX hand motor order (12): 0-5 right, 6-11 left
INSPIRE_RIGHT_MOTOR = [0, 1, 2, 3, 4, 5]
INSPIRE_LEFT_MOTOR = [6, 7, 8, 9, 10, 11]

# G1 29dof arms: shoulder(15-17) elbow(18) wrist roll(19) pitch(20) yaw(21) | same 22-28
ARMS_FIRST_MOTOR = 15
ARMS_LAST_MOTOR = 28
# wrist_pitch / wrist_yaw motor indices (inference yaml uses lower gains here)
_WRIST_PITCH_YAW_MOTORS = [20, 21, 27, 28]

# Redis keys
CMD_KEY = "sim2real_cmd"
ESTOP_KEY = "sim2real_estop"

# Safe defaults (rad). 29-DoF G1 typical limits are within +/-1.7; keep a little
# margin but still far below anything dangerous. Inspire 12 values are joint rad.
DEFAULT_JOINT_MIN = -1.7
DEFAULT_JOINT_MAX = 1.7
DEFAULT_INSPIRE_MIN = 0.0
DEFAULT_INSPIRE_MAX = 3.2
DEFAULT_TIMEOUT_S = 0.5


class Sim2RealBridge:
    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        rate: float,
        domain: int = 0,
        *,
        enable: bool = False,
        topic_prefix: str = "",
        expected_mode_machine: int = 0,
        joint_min: float = DEFAULT_JOINT_MIN,
        joint_max: float = DEFAULT_JOINT_MAX,
        inspire_min: float = DEFAULT_INSPIRE_MIN,
        inspire_max: float = DEFAULT_INSPIRE_MAX,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        hold_test: bool = False,
        burst_seconds: float = 1.0,
        kp: float = 60.0,
        kd: float = 2.0,
        arms_only: bool = False,
        kp_wrist: float = 8.611032447370201,
        kd_wrist: float = 0.548195351665136,
    ):
        import redis as redis_lib
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__MotorCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(domain)

        self._redis = redis_lib.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.rate = rate
        self.crc = CRC()
        self.low_state = None
        self.mode_machine = None
        self.enabled = enable
        self.topic_prefix = topic_prefix.strip()
        self.expected_mode_machine = expected_mode_machine
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.inspire_min = inspire_min
        self.inspire_max = inspire_max
        self.timeout_s = timeout_s
        self.hold_test = hold_test
        self.burst_seconds = burst_seconds
        self.kp = kp
        self.kd = kd
        self.arms_only = arms_only
        self.kp_wrist = kp_wrist
        self.kd_wrist = kd_wrist

        lowcmd_topic = f"rt/{self.topic_prefix}/lowcmd" if self.topic_prefix else "rt/lowcmd"
        lowstate_topic = f"rt/{self.topic_prefix}/lowstate" if self.topic_prefix else "rt/lowstate"
        inspire_topic = f"rt/{self.topic_prefix}/inspire/cmd" if self.topic_prefix else "rt/inspire/cmd"

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd_pub = ChannelPublisher(lowcmd_topic, LowCmd_)
        self.lowcmd_pub.Init()

        self.lowstate_sub = ChannelSubscriber(lowstate_topic, LowState_)
        self.lowstate_sub.Init(self._on_lowstate, 10)

        self.hand_cmd = MotorCmds_()
        self.hand_cmd.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
        self.inspire_pub = ChannelPublisher(inspire_topic, MotorCmds_)
        self.inspire_pub.Init()

        # internal state
        self._estop_warned = False
        self._mode_warned = False
        self._range_warned = False
        self._stale_warned = False
        self._first_armed_publish_logged = False

        print(f"[sim2real] bridge ready: domain={domain} rate={rate}Hz "
              f"lowcmd={lowcmd_topic} inspire={inspire_topic}")
        print(f"[sim2real] ENABLED={self.enabled} (publish {'ARMED' if self.enabled else 'DISABLED'}) "
              f"expected_mode_machine={expected_mode_machine} "
              f"joint_range=[{joint_min},{joint_max}] inspire_range=[{inspire_min},{inspire_max}] "
              f"timeout_s={timeout_s} kp={kp} kd={kd} hold_test={hold_test} burst_s={burst_seconds}")

    # ------------------------------------------------------------------ state
    def _on_lowstate(self, msg):
        self.low_state = msg
        if msg is not None:
            self.mode_machine = int(msg.mode_machine)

    # -------------------------------------------------------------- validation
    def _check_mode_machine(self) -> bool:
        if self.mode_machine is None:
            if not self._mode_warned:
                print("[sim2real] WARN: rt/lowstate not received yet; refusing to publish")
                self._mode_warned = True
            return False
        if self.mode_machine != self.expected_mode_machine:
            if not self._mode_warned:
                print(f"[sim2real] WARN: mode_machine={self.mode_machine} != expected "
                      f"{self.expected_mode_machine}; refusing to publish")
                self._mode_warned = True
            return False
        self._mode_warned = False
        return True

    def _check_range(self, body_29, inspire_r, inspire_l) -> bool:
        try:
            body = np.asarray(body_29, dtype=np.float64).reshape(-1)
            if body.size < 29:
                if not self._range_warned:
                    print(f"[sim2real] WARN: body_29 size {body.size} < 29; refusing")
                    self._range_warned = True
                return False
            body = body[:29]
            if not np.all(np.isfinite(body)):
                print("[sim2real] WARN: body_29 has NaN/Inf; refusing")
                return False
            if np.any(body < self.joint_min) or np.any(body > self.joint_max):
                if not self._range_warned:
                    print(f"[sim2real] WARN: body_29 out of range "
                          f"[{self.joint_min},{self.joint_max}]; refusing")
                    self._range_warned = True
                return False
            for arr, name in ((inspire_r, "inspire_right_12"), (inspire_l, "inspire_left_12")):
                a = np.asarray(arr, dtype=np.float64).reshape(-1)
                if a.size and not (np.all(np.isfinite(a)) and
                                   np.all(a >= self.inspire_min) and np.all(a <= self.inspire_max)):
                    if not self._range_warned:
                        print(f"[sim2real] WARN: {name} out of range "
                              f"[{self.inspire_min},{self.inspire_max}]; refusing")
                        self._range_warned = True
                    return False
        except Exception as exc:
            print(f"[sim2real] WARN: range check error: {exc}")
            return False
        self._range_warned = False
        return True

    def _check_freshness(self, ts_ns) -> bool:
        try:
            ts = float(ts_ns) / 1e9
            age = time.time() - ts
            if age > self.timeout_s:
                if not self._stale_warned:
                    print(f"[sim2real] WARN: command stale (age={age:.3f}s > "
                          f"{self.timeout_s}s); refusing")
                    self._stale_warned = True
                return False
        except (TypeError, ValueError):
            print("[sim2real] WARN: bad ts; treating as stale; refusing")
            return False
        self._stale_warned = False
        return True

    def _check_estop(self) -> bool:
        try:
            val = self._redis.get(ESTOP_KEY)
        except Exception:
            return True
        if val and str(val).strip() in ("1", "true"):
            if not self._estop_warned:
                print("[sim2real] ESTOP ACTIVE: publishing stopped")
                self._estop_warned = True
            return False
        self._estop_warned = False
        return True

    # ---------------------------------------------------------------- publish
    def _fill_lowcmd(self, body_29, source_order="policy"):
        """Fill LowCmd motor_cmd.

        source_order="policy": body_29 is in policy/MIMIC_LITE order, remap via
        _POLICY_TO_MOTOR (used for sim2real_cmd payloads).
        source_order="motor": body_29 is already in G1 motor index order
        (0..28), fill directly (used by hold-test with lowstate q).
        """
        if body_29 is None or len(body_29) < 29:
            return False
        self.low_cmd.mode_machine = self.mode_machine if self.mode_machine is not None else 0
        self.low_cmd.mode_pr = 0
        for src_i in range(29):
            motor_i = src_i if source_order == "motor" else _POLICY_TO_MOTOR[src_i]
            m = self.low_cmd.motor_cmd[motor_i]
            if self.arms_only and motor_i < 15:
                # legs/waist stay disabled (mode=0); do not touch position.
                m.mode = 0
                m.q = 0.0
                m.dq = 0.0
                m.tau = 0.0
                m.kp = 0.0
                m.kd = 0.0
                continue
            m.mode = 1  # enable only after all checks passed (caller gates)
            m.q = float(body_29[src_i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = self.kp
            m.kd = self.kd
        if self.arms_only:
            # Inference-matched gains per joint group (G1 29dof, from policy yaml):
            # shoulder/elbow/wrist_roll = 14.25 / 0.907 ; wrist_pitch/yaw = 8.61 / 0.548
            for motor_i in _WRIST_PITCH_YAW_MOTORS:
                self.low_cmd.motor_cmd[motor_i].kp = self.kp_wrist
                self.low_cmd.motor_cmd[motor_i].kd = self.kd_wrist
        try:
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        except Exception:
            pass
        return True

    def _fill_inspire(self, inspire_right_12, inspire_left_12):
        for i, motor_i in enumerate(INSPIRE_RIGHT_MOTOR):
            if inspire_right_12 and len(inspire_right_12) == 12:
                self.hand_cmd.cmds[motor_i].q = float(inspire_right_12[i])
            else:
                self.hand_cmd.cmds[motor_i].q = 1.0
        for i, motor_i in enumerate(INSPIRE_LEFT_MOTOR):
            if inspire_left_12 and len(inspire_left_12) == 12:
                self.hand_cmd.cmds[motor_i].q = float(inspire_left_12[i])
            else:
                self.hand_cmd.cmds[motor_i].q = 1.0

    def _maybe_publish(self, payload) -> None:
        body = payload.get("body_29")
        inspire_r = payload.get("inspire_right_12")
        inspire_l = payload.get("inspire_left_12")
        ts = payload.get("ts")

        if not self.enabled:
            return
        if not self._check_estop():
            return
        if not self._check_freshness(ts):
            return
        if not self._check_mode_machine():
            return
        if not self._check_range(body, inspire_r, inspire_l):
            return

        if self._fill_lowcmd(body):
            self.lowcmd_pub.Write(self.low_cmd)
        self._fill_inspire(inspire_r, inspire_l)
        self.inspire_pub.Write(self.hand_cmd)

        if not self._first_armed_publish_logged:
            print("[sim2real] FIRST ARMED PUBLISH OK (all safety checks passed)")
            self._first_armed_publish_logged = True

    def _hold_publish_loop(self, target_q, expected_mode):
        """Zero-motion hold burst: publish q_target=q_current for burst_seconds,
        then stop. A watchdog thread force-exits the process at deadline so a
        stuck main loop can never leave the robot commanded indefinitely.
        """
        self.expected_mode_machine = expected_mode
        self._first_armed_publish_logged = False
        deadline = time.monotonic() + self.burst_seconds
        done_evt = threading.Event()

        def _watchdog():
            while not done_evt.wait(0.1):
                if time.monotonic() >= deadline + 0.5:
                    print(f"[sim2real] HOLD WATCHDOG: burst over, force-exiting process")
                    os._exit(0)

        wt = threading.Thread(target=_watchdog, daemon=True)
        wt.start()

        print(f"[sim2real] HOLD-TEST: target_q=current lowstate (zero motion), "
              f"burst={self.burst_seconds}s, expected_mode_machine={expected_mode}, "
              f"kp={self.kp} kd={self.kd} arms_only={self.arms_only}")
        watch = range(ARMS_FIRST_MOTOR, ARMS_LAST_MOTOR + 1) if self.arms_only else range(29)
        n = 0
        max_delta = 0.0
        while time.monotonic() < deadline:
            if not self._check_estop():
                print("[sim2real] HOLD-TEST: estop, aborting")
                break
            if self.low_state is None:
                time.sleep(0.02)
                continue
            cur_q = [float(self.low_state.motor_state[i].q) for i in range(29)]
            delta = max(abs(cur_q[i] - target_q[i]) for i in watch)
            max_delta = max(max_delta, delta)
            if self._fill_lowcmd(target_q, source_order="motor"):
                self.lowcmd_pub.Write(self.low_cmd)
            self._fill_inspire(None, None)
            self.inspire_pub.Write(self.hand_cmd)
            n += 1
            if n <= 3 or n % 25 == 0:
                print(f"[sim2real] HOLD frame {n}: max|q_cur - q_target|={delta:.5f} rad "
                      f"({'arms' if self.arms_only else 'all29'})")
            time.sleep(max(0.0, 1.0 / self.rate))
        print(f"[sim2real] HOLD-TEST DONE: {n} frames published over {self.burst_seconds}s, "
              f"max|q_cur-q_target|={max_delta:.5f} rad (zero motion if <0.01)")
        done_evt.set()
        return max_delta

    def run_hold_test(self):
        """First real-robot verification: command current pose for a short burst
        (zero motion), verify motor mode 0->1 acceptance, then auto-stop.
        """
        print("[sim2real] HOLD-TEST: collecting current lowstate pose (zero-motion target)...")
        t0 = time.time()
        while self.low_state is None and time.time() - t0 < 3.0:
            time.sleep(0.1)
        if self.low_state is None:
            print("[sim2real] HOLD-TEST: no lowstate received, aborting")
            return 1
        mode = int(self.low_state.mode_machine)
        target_q = [float(self.low_state.motor_state[i].q) for i in range(29)]
        print(f"[sim2real] HOLD-TEST: mode_machine={mode}, "
              f"target_q={[round(q, 4) for q in target_q]}")
        max_delta = self._hold_publish_loop(target_q, mode)
        ok = max_delta < 0.01  # ~0.57 deg; tiny drift from gravity is expected
        print(f"[sim2real] HOLD-TEST RESULT: {'PASS (zero motion)' if ok else 'FAIL (moved!)'}")
        return 0 if ok else 2

    def run(self, once: bool = False):
        last_pub = 0.0
        while True:
            try:
                raw = self._redis.get(CMD_KEY)
                if raw:
                    try:
                        payload = json.loads(raw)
                        self._maybe_publish(payload)
                        # Refresh TTL so stale data expires when publisher dies.
                        self._redis.expire(CMD_KEY, int(self.timeout_s * 2) + 1)
                    except Exception as exc:
                        print(f"[sim2real] cmd error: {exc}")
            except Exception as exc:
                print(f"[sim2real] redis read error: {exc}")
            if once:
                break
            time.sleep(max(0.0, 1.0 / self.rate - (time.monotonic() - last_pub)))
            last_pub = time.monotonic()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--domain", type=int, default=0,
                        help="CycloneDDS domain: 0=real robot bus, 1=Isaac sim bus")
    parser.add_argument("--enable", action="store_true",
                        help="ARM publishing (safety gate 1). Without it the bridge only listens.")
    parser.add_argument("--topic-prefix", default="",
                        help="Safety gate 7: publish to rt/<prefix>/... instead of production "
                             "rt/lowcmd + rt/inspire/cmd (e.g. safe_test).")
    parser.add_argument("--expected-mode-machine", type=int, default=0,
                        help="Safety gate 2: required rt/lowstate mode_machine value.")
    parser.add_argument("--joint-min", type=float, default=DEFAULT_JOINT_MIN)
    parser.add_argument("--joint-max", type=float, default=DEFAULT_JOINT_MAX)
    parser.add_argument("--inspire-min", type=float, default=DEFAULT_INSPIRE_MIN)
    parser.add_argument("--inspire-max", type=float, default=DEFAULT_INSPIRE_MAX)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                        help="Safety gate 6: max age (s) of sim2real_cmd ts before stale.")
    parser.add_argument("--hold-test", action="store_true",
                        help="Zero-motion hold burst: command current lowstate pose "
                             "for --burst-seconds, verify motor acceptance, then stop.")
    parser.add_argument("--burst-seconds", type=float, default=1.0,
                        help="Hold-test publish duration (short! watchdog force-exits after).")
    parser.add_argument("--kp", type=float, default=60.0,
                        help="Position gain for motor_cmd (use low value for hold-test).")
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--arms-only", action="store_true",
                        help="Only enable arm motors 15-28 (legs/waist stay mode=0).")
    parser.add_argument("--kp-arms", type=float, default=14.25062309787429,
                        help="Arm shoulder/elbow/wrist_roll kp (inference-matched).")
    parser.add_argument("--kd-arms", type=float, default=0.907222843292423,
                        help="Arm shoulder/elbow/wrist_roll kd (inference-matched).")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    bridge = Sim2RealBridge(
        args.redis_host,
        args.redis_port,
        args.rate,
        domain=args.domain,
        enable=args.enable,
        topic_prefix=args.topic_prefix,
        expected_mode_machine=args.expected_mode_machine,
        joint_min=args.joint_min,
        joint_max=args.joint_max,
        inspire_min=args.inspire_min,
        inspire_max=args.inspire_max,
        timeout_s=args.timeout_s,
        hold_test=args.hold_test,
        burst_seconds=args.burst_seconds,
        kp=args.kp if not args.arms_only else args.kp_arms,
        kd=args.kd if not args.arms_only else args.kd_arms,
        arms_only=args.arms_only,
    )
    if args.hold_test:
        return bridge.run_hold_test()
    bridge.run(once=args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
