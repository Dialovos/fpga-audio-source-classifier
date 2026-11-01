"""
Make rumble, hiss, and short knocks or clicks before setting their levels.

This file has two groups:
    * Shared sound steps
    * Event sounds
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# ********************************
# Shared sound steps
# ********************************


# Returns a number drawn from a setting's range. It stops with a TypeError if
# rng is not a numpy Generator.
def _draw(settings, key, rng) -> float:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")

    return float(rng.uniform(*settings[key]))


# Returns the sound after applying soft fades in place at both ends.
def _fade(audio, fade_s, sample_rate) -> np.ndarray:
    count = min(round(fade_s * sample_rate), len(audio) // 2)
    fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, count))

    audio[:count] *= fade
    audio[-count:] *= fade[::-1]

    return audio


# ********************************
# Event sounds
# ********************************


# Returns low noise with optional hum and every drawn number in its recipe. It
# stops with a TypeError if rng is not a numpy Generator.
def rumble(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    duration = _draw(settings, "duration_s", rng)
    cutoff = _draw(settings, "cutoff_hz", rng)
    fade = _draw(settings, "fade_s", rng)

    audio = rng.standard_normal(round(duration * sample_rate))
    lowpass = signal.butter(2, cutoff, fs=sample_rate, output="sos")
    highpass = signal.butter(2, 15, btype="highpass", fs=sample_rate, output="sos")
    audio = signal.sosfilt(lowpass, audio)
    audio = signal.sosfilt(highpass, audio)

    recipe = {
        "duration_s": duration,
        "cutoff_hz": cutoff,
        "fade_s": fade,
        "hum": bool(rng.random() < 0.5),
    }

    if recipe["hum"]:
        frequency = _draw(settings, "hum_hz", rng)
        level = _draw(settings, "hum_db", rng)
        time = np.arange(len(audio)) / sample_rate
        peak = np.sqrt(2 * np.mean(audio**2)) * 10 ** (level / 20)
        audio += peak * np.sin(2 * np.pi * frequency * time)
        recipe.update({"hum_hz": frequency, "hum_db": level})

    return _fade(audio, fade, sample_rate), recipe


# Returns high noise with fades and every drawn number in its recipe. It stops
# with a TypeError if rng is not a numpy Generator.
def hiss(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    duration = _draw(settings, "duration_s", rng)
    cutoff = _draw(settings, "cutoff_hz", rng)
    fade = _draw(settings, "fade_s", rng)

    audio = rng.standard_normal(round(duration * sample_rate))
    highpass = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    lowpass = signal.butter(4, 20000, fs=sample_rate, output="sos")
    audio = signal.sosfilt(highpass, audio)
    audio = signal.sosfilt(lowpass, audio)

    recipe = {"duration_s": duration, "cutoff_hz": cutoff, "fade_s": fade}

    return _fade(audio, fade, sample_rate), recipe


# Returns a knock or click lasting at most 50 ms, with every drawn number in its
# recipe. It stops with a TypeError if rng is not a numpy Generator.
def transient(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")

    kind = "click"

    if rng.integers(0, 2) == 0:
        kind = "knock"

    decay = _draw(settings, f"{kind}_decay_s", rng)
    duration = min(5 * decay, 0.05)
    time = np.arange(round(duration * sample_rate)) / sample_rate
    recipe = {"kind": kind, "decay_s": decay, "duration_s": duration}

    if kind == "knock":
        frequency = _draw(settings, "freq_hz", rng)
        audio = np.sin(2 * np.pi * frequency * time)
        recipe["freq_hz"] = frequency
    else:
        audio = rng.standard_normal(len(time))

    return audio * np.exp(-time / decay), recipe
