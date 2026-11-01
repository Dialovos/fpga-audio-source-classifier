"""
Simple test sounds for checking the pipeline code.

These are smoke-test signals, not the project's event classes: a steady tone
(like a rumble at one pitch), a short low thump, and white noise. Every
function returns mono float64 audio with amplitudes relative to full scale 1.0.

This file has two groups:
    * Shared checks
    * The test sounds
"""

from __future__ import annotations

import numpy as np

# ********************************
# Shared checks
# ********************************


# Returns the time of every sample in a clip, in seconds (0, 1/rate, 2/rate, and
# so on). It stops with a ValueError if seconds or sample_rate isn't positive.
def _time_axis(seconds: float, sample_rate: float) -> np.ndarray:
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    n_samples = round(float(seconds) * float(sample_rate))

    return np.arange(n_samples) / sample_rate


# Stops unless the frequency is above 0 and below Nyquist (half the sample
# rate).
def _check_frequency(freq_hz: float, sample_rate: float) -> None:
    nyquist = sample_rate / 2

    if not 0 < freq_hz < nyquist:
        raise ValueError(f"freq_hz must be between 0 and {nyquist} Hz, got {freq_hz}")


# Stops unless the amplitude is above 0 and at most 1 (full scale).
def _check_amplitude(amplitude: float) -> None:
    if not 0 < amplitude <= 1:
        raise ValueError(f"amplitude must be in (0, 1], got {amplitude}")


# ********************************
# The test sounds
# ********************************


# Makes a steady sine tone, like a 100 Hz rumble. The pitch has to be between 0
# and Nyquist, and the amplitude (the peak level) has to be above 0 and up to 1.
# It returns mono audio, and stops with a ValueError if any setting is out of
# range.
def tone(freq_hz: float, seconds: float, sample_rate: float, amplitude: float = 0.5) -> np.ndarray:
    _check_frequency(freq_hz, sample_rate)
    _check_amplitude(amplitude)

    t = _time_axis(seconds, sample_rate)

    return amplitude * np.sin(2 * np.pi * freq_hz * t)


# Makes a short low knock: a sine wave that starts at onset_s and fades out
# quickly. The onset has to be inside the clip, and after decay_s seconds the
# knock is down to about 37% of its peak. It returns mono audio that's exactly
# zero before the onset, and stops with a ValueError if any setting is out of
# range.
def thump(
    seconds: float,
    sample_rate: float,
    onset_s: float = 0.1,
    freq_hz: float = 60.0,
    decay_s: float = 0.05,
    amplitude: float = 0.8,
) -> np.ndarray:
    _check_frequency(freq_hz, sample_rate)

    if decay_s <= 0:
        raise ValueError(f"decay_s must be positive, got {decay_s}")

    _check_amplitude(amplitude)

    t = _time_axis(seconds, sample_rate)

    # t[-1] is the time of the last sample, so the knock has to start before it.
    if not 0 <= onset_s < t[-1]:
        raise ValueError(f"onset_s must be inside the clip (0 to {seconds} s), got {onset_s}")

    out = np.zeros_like(t)

    after_onset = t >= onset_s
    elapsed = t[after_onset] - onset_s

    # The fade: starts at the full amplitude and shrinks over time.
    envelope = amplitude * np.exp(-elapsed / decay_s)

    out[after_onset] = envelope * np.sin(2 * np.pi * freq_hz * elapsed)

    return out


# Makes random noise (equal energy at every frequency) at an exact RMS level. It
# needs a seeded numpy random generator, like np.random.default_rng(seed), and a
# plain seed number isn't allowed, so every example uses its own generator. It
# returns mono audio. It stops with a TypeError if rng isn't a numpy random
# generator, and with a ValueError if the level or the length is bad.
def white_noise(seconds: float, sample_rate: float, rng: np.random.Generator, rms: float = 0.1) -> np.ndarray:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator, for example np.random.default_rng(seed)")

    if rms <= 0:
        raise ValueError(f"rms must be positive, got {rms}")

    n_samples = len(_time_axis(seconds, sample_rate))

    # Random numbers from a bell curve. On their own they average out to about RMS 1.
    noise = rng.standard_normal(n_samples)

    # Measure the actual RMS, then scale so it hits the requested level exactly.
    actual_rms = np.sqrt(np.mean(noise**2))

    return noise * (rms / actual_rms)
