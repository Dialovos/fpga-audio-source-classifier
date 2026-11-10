"""
Make the 7 event sounds before their levels get set.

Each one is a sound you'd hear in a cockpit, spread over low, middle, and high
frequencies so the channel filters bury them differently:
    * rumble: low noise below ~150 Hz
    * hum: a steady low tone with harmonics, 50-400 Hz
    * whine: a steady high tone, 2-12 kHz
    * hiss: high noise from 4-8 kHz up to 20 kHz
    * alarm: a beeping tone, 500 Hz-3 kHz, switching on and off
    * thump: a short low knock, a decaying 80 Hz-1 kHz tone
    * click: a short noise burst, cut off at 20 kHz like hiss

This file has two groups:
    * Shared sound steps
    * Event sounds
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# Every sound stops at 20 kHz, so none of them can be spotted by energy a real
# microphone wouldn't pick up.
TOP_HZ = 20000

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


# Returns a tone at freq_hz plus its harmonics up to count, each one 1/n as loud
# as the first, skipping any harmonic at or above TOP_HZ.
def _harmonic_tone(freq_hz, count, n_samples, sample_rate) -> np.ndarray:
    time = np.arange(n_samples) / sample_rate
    audio = np.zeros(n_samples)

    for harmonic in range(1, count + 1):
        if harmonic * freq_hz < TOP_HZ:
            audio += np.sin(2 * np.pi * harmonic * freq_hz * time) / harmonic

    return audio


# ********************************
# Event sounds
# ********************************


# Returns low noise and every drawn number in its recipe. It stops with a
# TypeError if rng is not a numpy Generator.
def rumble(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    duration = _draw(settings, "duration_s", rng)
    cutoff = _draw(settings, "cutoff_hz", rng)
    fade = _draw(settings, "fade_s", rng)

    audio = rng.standard_normal(round(duration * sample_rate))
    lowpass = signal.butter(2, cutoff, fs=sample_rate, output="sos")
    highpass = signal.butter(2, 15, btype="highpass", fs=sample_rate, output="sos")
    audio = signal.sosfilt(lowpass, audio)
    audio = signal.sosfilt(highpass, audio)

    recipe = {"duration_s": duration, "cutoff_hz": cutoff, "fade_s": fade}

    return _fade(audio, fade, sample_rate), recipe


# Returns a steady low tone with 4 harmonics and every drawn number in its
# recipe. It stops with a TypeError if rng is not a numpy Generator.
def hum(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    duration = _draw(settings, "duration_s", rng)
    frequency = _draw(settings, "freq_hz", rng)
    fade = _draw(settings, "fade_s", rng)

    audio = _harmonic_tone(frequency, 4, round(duration * sample_rate), sample_rate)
    recipe = {"duration_s": duration, "freq_hz": frequency, "fade_s": fade}

    return _fade(audio, fade, sample_rate), recipe


# Returns a steady high tone with one harmonic and every drawn number in its
# recipe. It stops with a TypeError if rng is not a numpy Generator.
def whine(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    duration = _draw(settings, "duration_s", rng)
    frequency = _draw(settings, "freq_hz", rng)
    fade = _draw(settings, "fade_s", rng)

    audio = _harmonic_tone(frequency, 2, round(duration * sample_rate), sample_rate)
    recipe = {"duration_s": duration, "freq_hz": frequency, "fade_s": fade}

    return _fade(audio, fade, sample_rate), recipe


# Returns high noise with fades and every drawn number in its recipe. It stops
# with a TypeError if rng is not a numpy Generator.
def hiss(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    duration = _draw(settings, "duration_s", rng)
    cutoff = _draw(settings, "cutoff_hz", rng)
    fade = _draw(settings, "fade_s", rng)

    audio = rng.standard_normal(round(duration * sample_rate))
    highpass = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    lowpass = signal.butter(4, TOP_HZ, fs=sample_rate, output="sos")
    audio = signal.sosfilt(highpass, audio)
    audio = signal.sosfilt(lowpass, audio)

    recipe = {"duration_s": duration, "cutoff_hz": cutoff, "fade_s": fade}

    return _fade(audio, fade, sample_rate), recipe


# Returns a beeping tone, with 3 harmonics, that switches on for beep_s and off
# for gap_s until it fills its duration, ending on a beep. Each beep gets its
# own short fades. It stops with a TypeError if rng is not a numpy Generator.
def alarm(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    duration = _draw(settings, "duration_s", rng)
    frequency = _draw(settings, "freq_hz", rng)
    beep_s = _draw(settings, "beep_s", rng)
    gap_s = _draw(settings, "gap_s", rng)
    fade = _draw(settings, "fade_s", rng)

    beep_samples = round(beep_s * sample_rate)
    gap_samples = round(gap_s * sample_rate)
    beeps = max(1, int((duration + gap_s) // (beep_s + gap_s)))
    n_samples = beeps * beep_samples + (beeps - 1) * gap_samples
    tone = _harmonic_tone(frequency, 3, n_samples, sample_rate)
    audio = np.zeros(n_samples)

    for beep in range(beeps):
        first = beep * (beep_samples + gap_samples)
        end = first + beep_samples
        audio[first:end] = _fade(tone[first:end].copy(), fade, sample_rate)

    recipe = {
        "duration_s": n_samples / sample_rate,
        "freq_hz": frequency,
        "beep_s": beep_s,
        "gap_s": gap_s,
        "beeps": beeps,
        "fade_s": fade,
    }

    return audio, recipe


# Returns a decaying low tone lasting at most 50 ms, with every drawn number in
# its recipe. It stops with a TypeError if rng is not a numpy Generator.
def thump(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    frequency = _draw(settings, "freq_hz", rng)
    decay = _draw(settings, "decay_s", rng)
    duration = min(5 * decay, 0.05)
    time = np.arange(round(duration * sample_rate)) / sample_rate

    audio = np.sin(2 * np.pi * frequency * time) * np.exp(-time / decay)
    recipe = {"freq_hz": frequency, "decay_s": decay, "duration_s": duration}

    return audio, recipe


# Returns a decaying noise burst lasting at most 50 ms, cut off at TOP_HZ, with
# every drawn number in its recipe. It stops with a TypeError if rng is not a
# numpy Generator.
def click(settings, sample_rate, rng) -> tuple[np.ndarray, dict]:
    decay = _draw(settings, "decay_s", rng)
    duration = min(5 * decay, 0.05)
    time = np.arange(round(duration * sample_rate)) / sample_rate

    noise = rng.standard_normal(len(time)) * np.exp(-time / decay)
    lowpass = signal.butter(4, TOP_HZ, fs=sample_rate, output="sos")
    audio = signal.sosfilt(lowpass, noise)
    recipe = {"decay_s": decay, "duration_s": duration}

    return audio, recipe
