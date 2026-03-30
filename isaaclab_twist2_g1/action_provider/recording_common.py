"""Shared asynchronous episode recording helpers."""

from __future__ import annotations

import copy
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np


class AsyncEpisodeRecorder:
    """Background NPZ episode recorder with pluggable frame organizer."""

    def __init__(
        self,
        *,
        save_dir: str,
        task_name: str,
        organize_fn: Callable[[list[dict[str, Any]], int], dict[str, Any]],
        max_frames: int = 10000,
    ) -> None:
        self.save_dir = save_dir
        self.task_name = task_name
        self.organize_fn = organize_fn
        self.max_frames = max_frames

        os.makedirs(save_dir, exist_ok=True)

        self.is_recording = False
        self.recording_buffer: list[dict[str, Any]] = []
        self.frame_count = 0

        self.save_queue: queue.Queue[Any] = queue.Queue(maxsize=10)
        self.thread_running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=False)
        self.save_thread.start()

    def _save_worker(self) -> None:
        while self.thread_running:
            try:
                task = self.save_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            data_buffer, timestamp_us, callback = task
            success = self._save_to_disk(data_buffer, timestamp_us)
            if callback is not None:
                callback(success)
            self.save_queue.task_done()

    def _save_to_disk(self, data_buffer: list[dict[str, Any]], timestamp_us: int) -> bool:
        temp_basename = f"{self.task_name}_{timestamp_us}_temp"
        filename = f"{self.task_name}_{timestamp_us}.npz"
        temp_filepath = os.path.join(self.save_dir, temp_basename)
        final_filepath = os.path.join(self.save_dir, filename)
        try:
            organized_data = self.organize_fn(data_buffer, timestamp_us)
            np.savez_compressed(temp_filepath, **organized_data)
            os.rename(temp_filepath + ".npz", final_filepath)
            print(
                f"[AsyncEpisodeRecorder] saved {len(data_buffer)} frames to {final_filepath}"
            )
            return True
        except Exception as exc:
            print(f"[AsyncEpisodeRecorder] save failed: {exc}")
            temp_npz = temp_filepath + ".npz"
            if os.path.exists(temp_npz):
                try:
                    os.remove(temp_npz)
                except OSError:
                    pass
            return False

    def start_recording(self) -> None:
        if self.is_recording:
            return
        self.is_recording = True
        self.recording_buffer = []
        self.frame_count = 0

    def add_frame(self, frame_data: dict[str, Any]) -> None:
        if not self.is_recording:
            return
        if self.frame_count >= self.max_frames:
            self.save_recording()
            return
        self.recording_buffer.append(copy.deepcopy(frame_data))
        self.frame_count += 1

    def save_recording(self, completion_callback: Callable[[bool], None] | None = None) -> None:
        if not self.is_recording:
            return
        if not self.recording_buffer:
            self.is_recording = False
            return
        self.is_recording = False
        timestamp_us = int(time.time() * 1_000_000)
        self.save_queue.put((self.recording_buffer, timestamp_us, completion_callback))
        self.recording_buffer = []
        self.frame_count = 0

    def cancel_recording(self) -> None:
        self.is_recording = False
        self.recording_buffer = []
        self.frame_count = 0

    def shutdown(self) -> None:
        if self.is_recording:
            self.cancel_recording()
        if not self.save_queue.empty():
            self.save_queue.join()
        self.thread_running = False
        self.save_queue.put(None)
        if self.save_thread.is_alive():
            self.save_thread.join(timeout=10.0)
