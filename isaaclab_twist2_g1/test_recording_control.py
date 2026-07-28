import tempfile
import unittest
from pathlib import Path

import numpy as np

from action_provider.recording_common import AsyncEpisodeRecorder
from action_provider.recording_control import (
    get_recording_control_identity,
    should_accept_recording_control,
)


class RecordingControlTests(unittest.TestCase):
    def test_reset_race_queues_only_one_episode(self):
        organized_frame_counts = []

        def organize(frames, _timestamp_us):
            organized_frame_counts.append(len(frames))
            return {"frame_index": np.arange(len(frames), dtype=np.int64)}

        with tempfile.TemporaryDirectory() as save_dir:
            recorder = AsyncEpisodeRecorder(
                save_dir=save_dir,
                task_name="mimic_lite_test",
                organize_fn=organize,
            )
            try:
                recorder.start_recording()
                for frame_index in range(3):
                    recorder.add_frame({"frame_index": frame_index})

                save_event = {
                    "command": "save_and_reset",
                    "sequence": 7,
                    "source": "twist2_teleop_server",
                }
                last_identity = None
                if should_accept_recording_control(
                    save_event,
                    last_identity=last_identity,
                    waiting_for_reset_complete=False,
                ):
                    last_identity = get_recording_control_identity(save_event)
                    recorder.save_recording()

                recorder.start_recording()
                recorder.add_frame({"frame_index": 0})
                if should_accept_recording_control(
                    save_event,
                    last_identity=last_identity,
                    waiting_for_reset_complete=False,
                ):
                    recorder.save_recording()
            finally:
                recorder.shutdown()

            self.assertEqual(organized_frame_counts, [3])
            self.assertEqual(len(list(Path(save_dir).glob("*.npz"))), 1)

    def test_reset_race_does_not_accept_a_second_save(self):
        last_identity = None
        save_count = 0
        events = [
            {
                "command": "save_and_reset",
                "sequence": 7,
                "source": "twist2_teleop_server",
                "timestamp_ms": 1_900,
            },
            # The publisher rewrites the held command after the new ready epoch.
            {
                "command": "save_and_reset",
                "sequence": 7,
                "source": "twist2_teleop_server",
                "timestamp_ms": 2_100,
            },
            {
                "command": "start",
                "sequence": 8,
                "source": "twist2_teleop_server",
                "timestamp_ms": 2_200,
            },
        ]

        for payload in events:
            if should_accept_recording_control(
                payload,
                last_identity=last_identity,
                waiting_for_reset_complete=False,
            ):
                last_identity = get_recording_control_identity(payload)
                save_count += payload["command"] == "save_and_reset"

        self.assertEqual(save_count, 1)

    def test_held_save_event_keeps_identity_across_reset(self):
        original = {
            "command": "save_and_reset",
            "sequence": 7,
            "source": "twist2_teleop_server",
            "timestamp_ms": 1_000,
        }
        rewritten_after_reset = {**original, "timestamp_ms": 2_100}

        consumed_identity = get_recording_control_identity(original)

        self.assertFalse(
            should_accept_recording_control(
                rewritten_after_reset,
                last_identity=consumed_identity,
                waiting_for_reset_complete=False,
            )
        )

    def test_legacy_payload_identity_uses_source(self):
        payload = {"source": "server", "sequence": "3"}

        self.assertEqual(get_recording_control_identity(payload), ("server", 3))


if __name__ == "__main__":
    unittest.main()
