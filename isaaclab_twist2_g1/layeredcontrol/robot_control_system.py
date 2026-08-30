# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
A layered robot control system
"""

import time
from typing import Optional, Dict, Any
import torch
from dataclasses import dataclass
from action_provider.action_base import ActionProvider


@dataclass
class ControlConfig:
    """minimal control configuration"""
    step_hz: int = 500  # the frequency of the low-level execution
    replay_mode: bool = False
    use_rl_action_mode: bool = False


class RobotController:
    """robot controller
    """

    def __init__(self, env, config: ControlConfig):
        self.env = env
        self.config = config
        self.action_provider: Optional[ActionProvider] = None
        self.is_running = False

        # minimal frequency control
        self._step_interval = 1.0 / config.step_hz
        self._last_step_time = 0.0

        all_joint_names = env.scene["robot"].data.joint_names
        self._last_action = torch.zeros(len(all_joint_names), device=env.device)

        # pre-calculate the sleep threshold (avoid calculating every time)
        self._sleep_threshold = 0.0002
        self._sleep_adjustment = 0.0001

        # minimal statistics
        self.step_count = 0
        self._start_time = 0.0

        # minimal performance analysis
        self._profile_counter = 0
        self._profile_interval = 2000  # reduce the printing frequency

        # cache the function reference (reduce the lookup overhead)
        self._perf_counter = time.perf_counter
        self._time_sleep = time.sleep

        print(f"  - control frequency: {config.step_hz}Hz")

    def set_action_provider(self, provider: ActionProvider):
        """set the action provider"""
        if self.action_provider:
            self.action_provider.stop()
            self.action_provider.cleanup()

        self.action_provider = provider
        self.env.action_provider = self.action_provider
        self.env.recovery_controller = self
        print(f"[SimpleController] set the action provider: {provider.name}")

    def capture_recovery_controller_state(self) -> dict[str, Any]:
        """Capture semantic controller state used by the next control step."""
        state = {
            "schema_version": 1,
            "controller_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "provider_type": (
                None
                if self.action_provider is None
                else f"{type(self.action_provider).__module__}.{type(self.action_provider).__qualname__}"
            ),
            "control_mode": {
                "replay_mode": bool(self.config.replay_mode),
                "use_rl_action_mode": bool(self.config.use_rl_action_mode),
            },
            "last_action": self._last_action.detach().clone(),
            "step_count": int(self.step_count),
        }
        self.preflight_restore_recovery_controller_state(state)
        return state

    def preflight_restore_recovery_controller_state(self, state: Any) -> None:
        """Validate controller recovery state without mutating the control loop."""
        expected_keys = {
            "schema_version",
            "controller_type",
            "provider_type",
            "control_mode",
            "last_action",
            "step_count",
        }
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("RobotController recovery state schema mismatch")
        if state["schema_version"] != 1:
            raise ValueError("RobotController recovery schema version mismatch")
        controller_type = f"{type(self).__module__}.{type(self).__qualname__}"
        if state["controller_type"] != controller_type:
            raise ValueError("RobotController recovery controller type mismatch")
        if self.action_provider is None:
            raise ValueError("RobotController recovery action_provider is unavailable")
        provider_type = (
            f"{type(self.action_provider).__module__}."
            f"{type(self.action_provider).__qualname__}"
        )
        if state["provider_type"] != provider_type:
            raise ValueError("RobotController recovery provider type mismatch")
        if getattr(self.env, "action_provider", None) is not self.action_provider:
            raise ValueError("RobotController recovery env.action_provider alias mismatch")
        if getattr(self.env, "recovery_controller", None) is not self:
            raise ValueError("RobotController recovery env.recovery_controller alias mismatch")
        expected_mode = {
            "replay_mode": bool(self.config.replay_mode),
            "use_rl_action_mode": bool(self.config.use_rl_action_mode),
        }
        if state["control_mode"] != expected_mode:
            raise ValueError("RobotController recovery control mode mismatch")
        last_action = state["last_action"]
        if not (
            isinstance(last_action, torch.Tensor)
            and last_action.layout == torch.strided
            and not last_action.is_quantized
            and last_action.shape == self._last_action.shape
            and last_action.dtype == self._last_action.dtype
            and last_action.device == self._last_action.device
            and bool(torch.isfinite(last_action).all())
        ):
            raise ValueError("RobotController recovery last_action schema mismatch")
        if type(state["step_count"]) is not int or state["step_count"] < 0:
            raise ValueError("RobotController recovery step_count schema mismatch")

    def restore_recovery_controller_state(self, state: Any) -> None:
        """Restore a preflighted controller fallback action and control cursor."""
        self.preflight_restore_recovery_controller_state(state)
        self._last_action.copy_(state["last_action"])
        self.step_count = int(state["step_count"])

    def start(self):
        """start the controller"""
        if self.is_running:
            return

        self.is_running = True
        self._start_time = time.time()
        self._last_step_time = self._perf_counter()

        # start the action provider
        if self.action_provider:
            self.action_provider.start()

        print("[SimpleController] the controller is started")

    def stop(self):
        """stop the controller"""
        if not self.is_running:
            return

        self.is_running = False

        if self.action_provider:
            self.action_provider.stop()

        print("[SimpleController] the controller is stopped")

    def step(self):
        """minimal control step - zero thread competition"""
        import time
        step_total_start = time.perf_counter()

        if not self.is_running:
            return

        # use the cached function reference
        perf_counter = self._perf_counter
        step_start = perf_counter()

        # 1. minimal action acquisition (synchronous, zero thread competition, pre-calculated strategy)
        action_start = perf_counter()
        action = None

        # try to get the action from the action provider
        if self.action_provider:
            action = self.action_provider.get_action(self.env)
            if action is not None:
                self._last_action = action
            # Replay / wholebody: physics runs inside the action provider (env.sim.step) and
            # env.step() is skipped below — reward_manager.compute must run here or sparse rewards
            # never update in sync with simulation.
            if self.config.replay_mode or self.config.use_rl_action_mode:
                from tools.get_reward import sync_reward_after_physics_step

                sync_reward_after_physics_step(self.env)

            # Replay rerecord jobs should stop the controller immediately once the
            # provider reports EOF completion, instead of continuing to replay the
            # last cached action indefinitely.
            if action is None:
                should_exit = getattr(self.action_provider, "should_exit_after_replay_complete", None)
                if callable(should_exit):
                    try:
                        if should_exit():
                            print("[SimpleController] replay completion acknowledged, stopping controller")
                            self.is_running = False
                            return
                    except Exception as exc:
                        print(f"[SimpleController] replay completion check failed: {exc}")

        # if no action is obtained, use the pre-calculated fallback strategy
        if action is None:
            action = self._last_action

        action_time = perf_counter() - action_start

        # 2. direct environment step
        env_start = perf_counter()
        if self.config.replay_mode or self.config.use_rl_action_mode:
            pass
            # self.env.sim.render()
        else:
            # Track env.step() frequency
            if not hasattr(self, '_env_step_count'):
                self._env_step_count = 0
                self._env_step_start_time = time.time()

            self._env_step_count += 1

            # Report env.step() frequency every 5 seconds
            if time.time() - self._env_step_start_time >= 5.0:
                elapsed = time.time() - self._env_step_start_time
                env_step_freq = self._env_step_count / elapsed
                print(f"[CONTROLLER]  env.step() frequency: {env_step_freq:.2f} Hz (count={self._env_step_count})")
                self._env_step_count = 0
                self._env_step_start_time = time.time()

            self.env.step(action)
        env_time = perf_counter() - env_start

        self.step_count += 1

        # 3. minimal frequency control (no rendering overhead, use the pre-calculated threshold)
        sleep_start = perf_counter()
        current_time = perf_counter()
        if self._last_step_time > 0:
            elapsed = current_time - self._last_step_time
            sleep_needed = self._step_interval - elapsed
            if sleep_needed > self._sleep_threshold:  # use the pre-calculated threshold
                self._time_sleep(sleep_needed - self._sleep_adjustment)  # use the pre-calculated adjustment value
        self._last_step_time = current_time
        sleep_time = perf_counter() - sleep_start

        step_total_time = time.perf_counter() - step_total_start

        # Track controller.step() frequency
        if not hasattr(self, '_controller_step_count'):
            self._controller_step_count = 0
            self._controller_step_start_time = time.time()

        self._controller_step_count += 1

        # Report controller.step() frequency every 5 seconds
        if time.time() - self._controller_step_start_time >= 5.0:
            elapsed = time.time() - self._controller_step_start_time
            controller_freq = self._controller_step_count / elapsed
            print(f"\n[CONTROLLER]  controller.step() frequency: {controller_freq:.2f} Hz (count={self._controller_step_count})")
            self._controller_step_count = 0
            self._controller_step_start_time = time.time()

        # if self.step_count % 30 == 0:
        #     print(f"\n[PERF] controller.step() breakdown:")
        #     print(f"  Action (get_action): {action_time * 1000:.2f}ms")
        #     print(f"  Env step: {env_time * 1000:.2f}ms")
        #     print(f"  Sleep: {sleep_time * 1000:.2f}ms")
        #     print(f"  Total: {step_total_time * 1000:.2f}ms")
        #     print(f"  Target interval: {self._step_interval * 1000:.2f}ms ({self.config.step_hz} Hz)")
        #     if step_total_time > 0:
        #         actual_hz = 1.0 / step_total_time
        #         print(f"  Actual frequency: {actual_hz:.2f} Hz")
        #         if actual_hz < self.config.step_hz * 0.8:
        #             print(f"  ⚠️  WARNING: Running {(1 - actual_hz/self.config.step_hz)*100:.1f}% slower than target!")
        #             # Debug: show what's taking time
        #             other_time = step_total_time - action_time - env_time - sleep_time
        #             print(f"  Debug: Action={action_time*1000:.2f}ms + Env={env_time*1000:.2f}ms + Sleep={sleep_time*1000:.2f}ms + Other={other_time*1000:.2f}ms")

        # 4. minimal performance print
        self._profile_counter += 1
        if self._profile_counter >= self._profile_interval:
            total_time = perf_counter() - step_start
            print(
                f"[Performance] A:{action_time * 1000:.1f}ms, E:{env_time * 1000:.1f}ms, S:{sleep_time * 1000:.1f}ms, T:{total_time * 1000:.1f}ms")
            self._profile_counter = 0

    def cleanup(self):
        """clean up the resources"""
        self.stop()
        if self.action_provider:
            self.action_provider.cleanup()

    def set_profiling(self, enabled: bool, interval: int = 2000):
        """set the performance analysis"""
        if enabled:
            self._profile_interval = interval
        else:
            self._profile_interval = 999999999  # actually disable the printing
