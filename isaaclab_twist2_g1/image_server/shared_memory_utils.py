# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
A simplified multi-image shared memory tool module
When writing, concatenate three images (head, left, right) horizontally and write them
When reading, split the concatenated image into three independent images
"""

import ctypes
import time
import numpy as np
import cv2
from multiprocessing import shared_memory
from typing import Optional, Dict, List
import struct

# shared memory configuration
SHM_NAME = "isaac_multi_image_shm"
# RGB: 640 * 480 * 3 * 4 cameras = 3,686,400 bytes
# Depth: 640 * 480 * 4 (float32) * 4 cameras = 4,915,200 bytes
# Total: ~8.6MB + 2KB header
SHM_SIZE = 640 * 480 * 3 * 4 + 640 * 480 * 4 * 4 + 2048


# define the enhanced header structure with depth support
class SimpleImageHeader(ctypes.Structure):
    """Enhanced image header structure with depth support"""
    _fields_ = [
        ('timestamp', ctypes.c_uint64),  # timestamp
        ('height', ctypes.c_uint32),  # image height
        ('width', ctypes.c_uint32),  # total width after concatenation
        ('channels', ctypes.c_uint32),  # number of channels (RGB)
        ('single_width', ctypes.c_uint32),  # single image width
        ('image_count', ctypes.c_uint32),  # number of images
        ('rgb_data_size', ctypes.c_uint32),  # RGB data size
        ('depth_data_size', ctypes.c_uint32),  # depth data size
        ('has_depth', ctypes.c_uint32),  # whether depth data is included (0 or 1)
    ]


class MultiImageWriter:
    """A simplified multi-image shared memory writer"""

    def __init__(self, shm_name: str = SHM_NAME, shm_size: int = SHM_SIZE):
        """Initialize the multi-image shared memory writer

        Args:
            shm_name: the name of the shared memory
            shm_size: the size of the shared memory
        """
        self.shm_name = shm_name
        self.shm_size = shm_size
        self._created = False  # Track if we created the shared memory

        try:
            # try to open the existing shared memory
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self._created = False
        except FileNotFoundError:
            # if not exist, create a new shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
            self._created = True
            # Initialize with zeros to avoid reading garbage data
            self.shm.buf[:] = bytes(shm_size)

        print(f"[MultiImageWriter] Shared memory initialized: {shm_name} (created={self._created})")

    def write_images(self, images: Dict[str, np.ndarray], depths: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """Write multiple RGB images and optional depth maps to shared memory

        Args:
            images: RGB image dictionary, key is camera name ('head', 'world', 'left', 'right')
            depths: Optional depth map dictionary, key is camera name with '_depth' suffix

        Returns:
            bool: whether the writing is successful
        """
        if not images or self.shm is None:
            return False

        try:
            # Process RGB images
            frames_to_concat = []
            depth_frames_to_concat = []
            image_order = ['head', 'world', 'left', 'right']

            for image_name in image_order:
                if image_name in images:
                    # Process RGB
                    image = images[image_name]
                    if not image.flags['C_CONTIGUOUS']:
                        image = np.ascontiguousarray(image)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    frames_to_concat.append(image)

                    # Process depth if available
                    if depths and image_name in depths:
                        depth = depths[image_name]
                        if not depth.flags['C_CONTIGUOUS']:
                            depth = np.ascontiguousarray(depth)
                        # Ensure depth is float32
                        if depth.dtype != np.float32:
                            depth = depth.astype(np.float32)
                        # Squeeze depth to 2D if it has shape (H, W, 1)
                        if len(depth.shape) == 3 and depth.shape[2] == 1:
                            depth = depth.squeeze(axis=2)
                        depth_frames_to_concat.append(depth)

            if not frames_to_concat:
                return False

            # Concatenate RGB images
            if len(frames_to_concat) > 1:
                concatenated_image = cv2.hconcat(frames_to_concat)
            else:
                concatenated_image = frames_to_concat[0]

            height, total_width, channels = concatenated_image.shape
            single_width = total_width // len(frames_to_concat)
            rgb_data_size = height * total_width * channels

            # Concatenate depth maps if available
            depth_data_size = 0
            concatenated_depth = None
            if depth_frames_to_concat:
                if len(depth_frames_to_concat) > 1:
                    concatenated_depth = np.hstack(depth_frames_to_concat)
                else:
                    concatenated_depth = depth_frames_to_concat[0]
                depth_data_size = concatenated_depth.nbytes

            # Prepare header
            header = SimpleImageHeader()
            header.timestamp = int(time.time() * 1000)
            header.height = height
            header.width = total_width
            header.channels = channels
            header.single_width = single_width
            header.image_count = len(frames_to_concat)
            header.rgb_data_size = rgb_data_size
            header.depth_data_size = depth_data_size
            header.has_depth = 1 if concatenated_depth is not None else 0

            # Write header
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_bytes = ctypes.string_at(ctypes.byref(header), header_size)
            self.shm.buf[:header_size] = header_bytes

            # Write RGB data
            rgb_offset = header_size
            rgb_bytes = concatenated_image.tobytes()
            self.shm.buf[rgb_offset:rgb_offset + len(rgb_bytes)] = rgb_bytes

            # Write depth data if available
            if concatenated_depth is not None:
                depth_offset = rgb_offset + len(rgb_bytes)
                depth_bytes = concatenated_depth.tobytes()
                self.shm.buf[depth_offset:depth_offset + len(depth_bytes)] = depth_bytes

            return True

        except Exception as e:
            print(f"[MultiImageWriter] Error writing to shared memory: {e}")
            print(f"Images: {list(images.keys())}, Depths: {list(depths.keys()) if depths else 'None'}")
            import traceback
            traceback.print_exc()
            return False

    def close(self):
        """Close the shared memory (but don't unlink it yet)"""
        if hasattr(self, 'shm') and self.shm is not None:
            try:
                self.shm.close()
                print(f"[MultiImageWriter] Shared memory closed: {self.shm_name}")
            except Exception as e:
                print(f"[MultiImageWriter] Error closing shared memory: {e}")

    def unlink(self):
        """Unlink (delete) the shared memory. Only call this if you created it."""
        if hasattr(self, 'shm') and self.shm is not None and self._created:
            try:
                self.shm.unlink()
                print(f"[MultiImageWriter] Shared memory unlinked: {self.shm_name}")
            except Exception as e:
                print(f"[MultiImageWriter] Error unlinking shared memory: {e}")

    def cleanup(self):
        """Close and unlink the shared memory (full cleanup)"""
        self.close()
        self.unlink()

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection"""
        try:
            self.cleanup()
        except Exception:
            pass  # Silently ignore errors during cleanup in __del__


class MultiImageReader:
    """A simplified multi-image shared memory reader"""

    def __init__(self, shm_name: str = SHM_NAME):
        """Initialize the multi-image shared memory reader

        Args:
            shm_name: the name of the shared memory
        """
        self.shm_name = shm_name
        self.last_timestamp = 0
        self.buffer = {}  # Always initialize as dict

        try:
            # open the shared memory
            self.shm = shared_memory.SharedMemory(name=shm_name)
            print(f"[MultiImageReader] Shared memory opened: {shm_name}")
        except FileNotFoundError:
            print(f"[MultiImageReader] Shared memory {shm_name} not found, will retry when reading")
            self.shm = None

    def read_images(self) -> Dict[str, np.ndarray]:
        """Read multiple RGB images and depth maps from shared memory

        Returns:
            Dict[str, np.ndarray]: Dictionary containing RGB images and depth maps.
                                   RGB images: 'head', 'world', 'left', 'right'
                                   Depth maps: 'head_depth', 'world_depth', etc.
                                   Returns empty dict or cached buffer if reading fails.
        """
        # Try to reconnect if shared memory is not available
        if self.shm is None:
            try:
                self.shm = shared_memory.SharedMemory(name=self.shm_name)
                print(f"[MultiImageReader] Reconnected to shared memory: {self.shm_name}")
            except FileNotFoundError:
                # Still not available, return empty buffer
                return self.buffer if isinstance(self.buffer, dict) else {}

        try:
            # Read header
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_data = bytes(self.shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)

            # Validate header data
            if header.timestamp == 0 or header.width == 0 or header.height == 0:
                # Shared memory not yet initialized by writer
                return self.buffer

            # Check for new data
            if header.timestamp <= self.last_timestamp:
                return self.buffer

            # Validate image count
            if header.image_count == 0 or header.image_count > 4:
                print(f"[MultiImageReader] Invalid image count: {header.image_count}")
                return self.buffer

            # Read RGB data
            rgb_offset = header_size
            rgb_end = rgb_offset + header.rgb_data_size

            # Validate RGB data size
            if header.rgb_data_size == 0 or rgb_end > len(self.shm.buf):
                print(f"[MultiImageReader] Invalid RGB data size: {header.rgb_data_size}")
                return self.buffer

            rgb_data = bytes(self.shm.buf[rgb_offset:rgb_end])

            concatenated_image = np.frombuffer(rgb_data, dtype=np.uint8)
            expected_size = header.height * header.width * header.channels

            if concatenated_image.size != expected_size:
                print(f"[MultiImageReader] RGB size mismatch: expected {expected_size}, got {concatenated_image.size}")
                return self.buffer

            concatenated_image = concatenated_image.reshape(header.height, header.width, header.channels)

            # Split RGB images
            images = {}  # Ensure images is always a dict
            image_names = ['head', 'world', 'left', 'right']
            single_width = header.single_width

            # Validate single_width
            if single_width == 0 or single_width * header.image_count != header.width:
                print(f"[MultiImageReader] Invalid single_width: {single_width}, image_count: {header.image_count}, total width: {header.width}")
                return self.buffer

            for i in range(header.image_count):
                if i < len(image_names):
                    start_col = i * single_width
                    end_col = start_col + single_width
                    single_image = concatenated_image[:, start_col:end_col, :]
                    images[image_names[i]] = single_image

            # Read depth data if available
            if header.has_depth and header.depth_data_size > 0:
                depth_offset = rgb_end
                depth_end = depth_offset + header.depth_data_size
                depth_data = bytes(self.shm.buf[depth_offset:depth_end])

                concatenated_depth = np.frombuffer(depth_data, dtype=np.float32)
                # Note: header.width is already the total concatenated width (single_width * image_count)
                expected_depth_size = header.height * header.width

                if concatenated_depth.size != expected_depth_size:
                    print(
                        f"[MultiImageReader] Depth size mismatch: expected {expected_depth_size}, got {concatenated_depth.size}")
                else:
                    concatenated_depth = concatenated_depth.reshape(header.height, header.width)

                    # Split depth maps
                    for i in range(header.image_count):
                        if i < len(image_names):
                            start_col = i * single_width
                            end_col = start_col + single_width
                            single_depth = concatenated_depth[:, start_col:end_col]
                            images[f"{image_names[i]}_depth"] = single_depth

            # Update buffer and timestamp
            self.buffer = images
            self.last_timestamp = header.timestamp
            return images

        except Exception as e:
            print(f"[MultiImageReader] Error reading from shared memory: {e}")
            import traceback
            traceback.print_exc()
            return self.buffer  # Return cached buffer on error

    def read_concatenated_image(self, only_head: bool = False) -> Optional[np.ndarray]:
        """Read the concatenated image (without splitting)

        Args:
            only_head: if True, only return the head camera image (first image in concatenation)

        Returns:
            np.ndarray: the concatenated image array (or only head camera if only_head=True); if the reading fails, return None
        """
        if self.shm is None:
            return None

        try:
            # read the header data
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_data = bytes(self.shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)

            # check if there is new data
            if header.timestamp <= self.last_timestamp:
                return None

            # read the concatenated image data
            start_offset = header_size
            end_offset = start_offset + header.rgb_data_size
            image_data = bytes(self.shm.buf[start_offset:end_offset])

            # convert to numpy array
            concatenated_image = np.frombuffer(image_data, dtype=np.uint8)

            # ensure the data size is correct
            expected_size = header.height * header.width * header.channels
            if concatenated_image.size != expected_size:
                print(f"[MultiImageReader] Data size mismatch: expected {expected_size}, got {concatenated_image.size}")
                return None

            # reshape the array
            concatenated_image = concatenated_image.reshape(header.height, header.width, header.channels)

            # update the timestamp
            self.last_timestamp = header.timestamp

            # If only_head is True, extract only the head camera (first image)
            if only_head and header.image_count > 1 and header.single_width > 0:
                # Extract the first single_width columns (head camera)
                head_image = concatenated_image[:, :header.single_width, :]
                return head_image

            # print(f"concatenated_image: {concatenated_image.shape}")
            return concatenated_image

        except Exception as e:
            print(f"[MultiImageReader] Error reading concatenated image from shared memory: {e}")
            return None

    def close(self):
        """Close the shared memory"""
        if self.shm is not None:
            self.shm.close()
            print(f"[MultiImageReader] Shared memory closed: {self.shm_name}")


# backward compatible class (single image)
class SharedMemoryWriter:
    """Backward compatible single image writer"""

    def __init__(self, shm_name: str = SHM_NAME, shm_size: int = SHM_SIZE):
        self.multi_writer = MultiImageWriter(shm_name, shm_size)

    def write_image(self, image: np.ndarray) -> bool:
        """Write a single image (as the head image)"""
        return self.multi_writer.write_images({'head': image})

    def close(self):
        self.multi_writer.close()


class SharedMemoryReader:
    """Backward compatible single image reader"""

    def __init__(self, shm_name: str = SHM_NAME):
        self.multi_reader = MultiImageReader(shm_name)

    def read_image(self) -> Optional[np.ndarray]:
        """Read a single image (the head image)"""
        images = self.multi_reader.read_images()
        return images.get('head') if images else None

    def close(self):
        self.multi_reader.close() 