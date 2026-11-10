"""
Measure frequency bands and their changes on centered audio frames.

This file has four groups:
    * Frame times
    * Window measurements
    * Band levels and changes
    * Frame features and context
"""

from __future__ import annotations

import numpy as np
from scipy import signal

SMALLEST_POWER = 1e-12

# ********************************
# Frame times
# ********************************


# Returns the number of frame centers inside the audio. It stops with a
# ValueError if the sample count or hop is not positive.
def frame_count(n_samples, hop) -> int:
    if n_samples < 1 or hop < 1:
        raise ValueError("n_samples and hop must be positive")

    return 1 + (n_samples - 1) // hop


# Returns each frame's center time in seconds, starting at zero.
def frame_times(n_frames, sample_rate, hop) -> np.ndarray:
    return np.arange(n_frames) * hop / sample_rate


# ********************************
# Window measurements
# ********************************


# Returns each band's summed power. If a band is narrower than one frequency
# bin, it uses the bin nearest the middle of that band.
def _bands(power, frequencies, edges) -> np.ndarray:
    result = np.empty((len(power), len(edges) - 1))

    for band in range(len(edges) - 1):
        first = np.searchsorted(frequencies, edges[band])
        end = np.searchsorted(frequencies, edges[band + 1])

        if first == end:
            center = np.sqrt(edges[band] * edges[band + 1])
            nearest = np.argmin(np.abs(frequencies - center))
            result[:, band] = power[:, nearest]
        else:
            result[:, band] = np.sum(power[:, first:end], axis=1)

    return result


# Returns band power, total power, and flatness from centered windows. At most
# 256 frames are measured together so the full window matrix is not copied.
def _measure(padded, padding, n_frames, sample_rate, settings, prefix):
    size = settings[f"{prefix}_window"]
    hop = settings["hop"]

    # Make a view of centered windows, then select one every hop samples.
    first_sample = padding - size // 2
    all_windows = np.lib.stride_tricks.sliding_window_view(padded[first_sample:], size)
    windows = all_windows[::hop]

    # Set up the window weights, frequency bins, and band edges once.
    window = signal.get_window("hann", size)
    frequencies = np.fft.rfftfreq(size, 1 / sample_rate)
    low_hz = settings[f"{prefix}_min_hz"]
    high_hz = settings[f"{prefix}_max_hz"]
    n_bands = settings[f"{prefix}_bands"]
    edges = np.geomspace(low_hz, high_hz, n_bands + 1)
    scale = size * np.sum(window**2)

    # Reserve only the small measurements kept for each frame.
    band_power = np.empty((n_frames, n_bands))
    total_power = np.empty(n_frames)
    flatness = np.empty(n_frames)
    selected = (frequencies >= 200) & (frequencies <= 24000)

    for first in range(0, n_frames, 256):
        end = min(first + 256, n_frames)

        # Apply the Hann weights and measure power at each frequency.
        weighted = windows[first:end] * window
        spectrum = np.fft.rfft(weighted, axis=1)
        power = np.abs(spectrum) ** 2 / scale
        power[:, 1:-1] *= 2
        del weighted
        del spectrum

        # Sum the power inside each band and over the whole window.
        band_power[first:end] = _bands(power, frequencies, edges)
        total_power[first:end] = np.sum(power, axis=1)

        # Compare the geometric and arithmetic means to measure how noise-like
        # the high spectrum is. The floor keeps silence finite.
        floored = np.maximum(power[:, selected], SMALLEST_POWER)
        geometric_mean = np.exp(np.mean(np.log(floored), axis=1))
        arithmetic_mean = np.mean(floored, axis=1)
        flatness[first:end] = geometric_mean / arithmetic_mean
        del power

    return band_power, total_power, flatness


# ********************************
# Band levels and changes
# ********************************


# Returns each band's share of the frame's power in dB. Dividing before taking
# the log keeps a quiet copy's spectral shape the same as a loud copy's shape.
# Silence stays finite and no part of the WAV is assumed to be background.
def band_levels(power, total_power) -> np.ndarray:
    fractions = power / np.maximum(total_power[:, None], 1e-30)

    return 10 * np.log10(np.maximum(fractions, SMALLEST_POWER))


# Returns each band's average over about 200 ms around a frame. This steadies
# quiet, lasting sounds. Existing frames at the WAV's edges may contain events
# just like any other part of the recording.
def band_averages(bands_db, sample_rate, hop) -> np.ndarray:
    half = max(1, round(0.1 * sample_rate / hop))
    averages = np.empty_like(bands_db)

    for frame in range(len(bands_db)):
        first = max(0, frame - half)
        end = min(len(bands_db), frame + half + 1)
        averages[frame] = np.mean(bands_db[first:end], axis=0)

    return averages


# Returns each band's change from its average over the previous 100 ms. This
# describes an attack or decay without depending on a recording's level. The
# first frame has zero change because it has no earlier frames.
def band_changes(bands_db, sample_rate, hop) -> np.ndarray:
    previous_frames = max(1, round(0.1 * sample_rate / hop))
    changes = np.zeros_like(bands_db)

    for frame in range(1, len(bands_db)):
        first = max(0, frame - previous_frames)
        previous = np.mean(bands_db[first:frame], axis=0)
        changes[frame] = bands_db[frame] - previous

    return changes


# Returns each band's spread (standard deviation) over about 500 ms around a
# frame. A beeping alarm swings up and down that fast, while a steady whine or
# hum stays flat. The neighboring frames only reach 30 ms either side.
def band_spreads(bands_db, sample_rate, hop) -> np.ndarray:
    half = max(1, round(0.25 * sample_rate / hop))
    spreads = np.empty_like(bands_db)

    for frame in range(len(bands_db)):
        first = max(0, frame - half)
        end = min(len(bands_db), frame + half + 1)
        spreads[frame] = np.std(bands_db[first:end], axis=0)

    return spreads


# Returns differences between neighboring short bands. These describe the
# spectrum's shape without changing when the whole signal gets louder.
def band_slopes(bands_db) -> np.ndarray:
    return np.diff(bands_db, axis=1)


# ********************************
# Frame features and context
# ********************************


# Returns the same float32 features and center times for every event type. It
# stops with a ValueError if the audio is empty, non-finite, or not mono.
def frame_features(audio, sample_rate, settings) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(audio, dtype=np.float64)

    if samples.ndim != 1 or len(samples) == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("audio must be non-empty finite mono audio")

    # Mirror the WAV at its ends so every centered window has its full length.
    count = frame_count(len(samples), settings["hop"])
    padding = settings["long_window"] // 2
    padded = np.pad(samples, (padding, padding), mode="reflect")

    # Measure low and high bands at exactly the same frame centers.
    long_power, long_total, long_flatness = _measure(
        padded, padding, count, sample_rate, settings, "long"
    )
    del long_flatness

    short_power, total, flatness = _measure(
        padded, padding, count, sample_rate, settings, "short"
    )
    del padded

    # Express band shares in dB. The overall level is kept in its own column,
    # so classifiers can separate a shape change from the signal getting louder.
    long_db = band_levels(long_power, long_total)
    short_db = band_levels(short_power, total)
    level_db = 10 * np.log10(np.maximum(total, SMALLEST_POWER))

    # Flux measures positive short-band changes since the previous frame.
    flux = np.zeros(count)
    raw_short_db = 10 * np.log10(np.maximum(short_power, SMALLEST_POWER))
    jumps = np.maximum(np.diff(raw_short_db, axis=0), 0)
    flux[1:] = np.sum(jumps, axis=1)

    # Keep the original measurements first, then add general time averages,
    # changes, and spreads. Every classifier receives all of these columns.
    averages = band_averages(short_db, sample_rate, settings["hop"])
    changes = band_changes(short_db, sample_rate, settings["hop"])
    slopes = band_slopes(short_db)
    spreads = band_spreads(short_db, sample_rate, settings["hop"])
    columns = [
        long_db,
        short_db,
        level_db,
        flatness,
        flux,
        averages,
        changes,
        slopes,
        spreads,
    ]
    features = np.column_stack(columns).astype(np.float32)
    times = frame_times(count, sample_rate, settings["hop"])

    return features, times


# Returns neighboring frame columns from left to right, repeating the edge rows
# within this clip. It stops with a ValueError if features has no rows or
# context is not a non-negative integer.
def add_context(features, context) -> np.ndarray:
    if features.ndim != 2 or len(features) == 0 or not isinstance(context, int) or context < 0:
        raise ValueError("features must have rows and context must be a non-negative integer")

    # Choose neighboring rows, stopping at this clip's first and last frames.
    rows = np.arange(len(features))
    columns = []

    for offset in range(-context, context + 1):
        selected = np.clip(rows + offset, 0, len(features) - 1)
        columns.append(features[selected])

    # Put the left neighbors first and the right neighbors last.
    return np.hstack(columns).astype(np.float32)
