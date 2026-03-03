# Copyright (c) 2025, HumanoidArena Project
# Adapted from instinctlab perlin noise utilities
# License: Apache License, Version 2.0
"""
Perlin noise generation utilities for terrain generation.
"""

import numpy as np
from typing import Sequence


def generate_perlin_noise_2d(shape: Sequence[int], res: Sequence[int]) -> np.ndarray:
    """Generate 2D Perlin noise.
    
    Args:
        shape: Shape of the output array (height, width).
        res: Resolution of the noise grid.
        
    Returns:
        2D numpy array with Perlin noise values in range [0, 1].
    """
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5


def generate_fractal_noise_2d(
    xSize: float = 20,
    ySize: float = 20,
    xSamples: int = 1600,
    ySamples: int = 1600,
    frequency: int = 10,
    fractalOctaves: int = 2,
    fractalLacunarity: float = 2.0,
    fractalGain: float = 0.25,
    zScale: float = 0.23,
    centering: bool = False,
) -> np.ndarray:
    """Generate 2D fractal noise using multiple octaves of Perlin noise.
    
    Args:
        xSize: Physical size along x-axis.
        ySize: Physical size along y-axis.
        xSamples: Number of samples along x-axis.
        ySamples: Number of samples along y-axis.
        frequency: Base frequency of the noise.
        fractalOctaves: Number of noise octaves.
        fractalLacunarity: Frequency multiplier between octaves.
        fractalGain: Amplitude multiplier between octaves.
        zScale: Vertical scale of the noise.
        centering: If True, center the noise around 0.
        
    Returns:
        2D numpy array with fractal noise values.
    """
    xScale = int(frequency * xSize)
    yScale = int(frequency * ySize)
    amplitude = 1

    # Ensure sample shape is compatible with scale
    expected_xSamples = int(xScale * (fractalLacunarity**fractalOctaves))
    expected_ySamples = int(yScale * (fractalLacunarity**fractalOctaves))

    # Use larger of expected or requested samples
    actual_xSamples = max(xSamples, expected_xSamples)
    actual_ySamples = max(ySamples, expected_ySamples)

    noise = np.zeros((actual_xSamples, actual_ySamples))
    for _ in range(fractalOctaves):
        noise += amplitude * generate_perlin_noise_2d(noise.shape, (xScale, yScale)) * zScale
        amplitude *= fractalGain
        xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

    if centering:
        noise -= np.mean(noise)

    return noise[:xSamples, :ySamples].copy()
