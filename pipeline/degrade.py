"""
Realistic damage: added noise and rounding to a bit depth.

Both are settings, not fixed steps. The proposal adds noise at several SNRs and
tests it both before and after the filter, so every step takes a stage
("before_filter" or "after_filter") and puts it in its record. The generator
decides the order. Each function returns (audio, record) so the example's record
says exactly what was done and where.

This file has three groups:
    * Shared checks
    * Added noise
    * Rounding to a bit depth
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

STAGES = ("before_filter", "after_filter")

MIN_BITS = 2
MAX_BITS = 32


# ********************************
# Shared checks
# ********************************


# Returns the audio as a numpy array of decimal numbers, or stops if it isn't
# mono.
def _mono(audio) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float64)

    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {samples.shape}")

    return samples


# Stops unless the stage is "before_filter" or "after_filter".
def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")


# Stops unless rng is a numpy random generator. A plain seed number isn't
# allowed.
def _check_rng(rng) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator, for example np.random.default_rng(seed)")


# Returns the bit depth as a whole number, or stops unless it's from 2 to 32.
def _check_bits(bits) -> int:
    # Python treats True as the number 1, so True and False are ruled out on purpose.
    # 24.0 is ruled out too, since it's probably a typo.
    is_whole_number = isinstance(bits, Integral) and not isinstance(bits, bool)

    if not is_whole_number or not MIN_BITS <= bits <= MAX_BITS:
        raise ValueError(f"bits must be an integer from {MIN_BITS} to {MAX_BITS}, got {bits!r}")

    return int(bits)


# ********************************
# Added noise
# ********************************


# Returns noisy audio and its record, using the input power or reference_power.
# It stops with a ValueError for an invalid reference, silent input without a
# reference, non-mono audio, or an unknown stage. A wrong rng raises TypeError.
def add_noise(
    audio, snr_db: float, rng: np.random.Generator, *, stage: str, reference_power=None
) -> tuple[np.ndarray, dict]:
    _check_rng(rng)
    _check_stage(stage)
    samples = _mono(audio)

    signal_power = np.mean(samples**2)

    if reference_power is not None:
        is_number = isinstance(reference_power, Real) and not isinstance(reference_power, bool)

        if not is_number or not np.isfinite(reference_power) or reference_power <= 0:
            raise ValueError("reference_power must be a positive finite number")

        signal_power = float(reference_power)

    # Silence has no power, so there's nothing to compare the noise to.
    if signal_power == 0:
        raise ValueError("can't set an SNR on silent audio")

    # An SNR of X dB means the noise power is the signal power divided by 10^(X/10).
    noise_rms = np.sqrt(signal_power / 10 ** (snr_db / 10))

    # Make random noise, then scale it so its level is exactly noise_rms
    # (random noise is only close to the right level on its own).
    noise = rng.standard_normal(len(samples))
    actual_rms = np.sqrt(np.mean(noise**2))
    noise *= noise_rms / actual_rms

    record = {
        "noise": "white gaussian",
        "stage": stage,
        "snr_db": float(snr_db),
        "noise_rms": float(noise_rms),
        "snr_reference": "input",
    }

    if reference_power is not None:
        record["snr_reference"] = "given"
        record["reference_power"] = float(reference_power)

    return samples + noise, record


# ********************************
# Rounding to a bit depth
# ********************************


# Rounds the audio like a recorder's converter, and clips anything too loud. A
# 24-bit converter can only store whole numbers ("codes") from -2^23 up to 2^23
# - 1, and each code is one "step" of level, where step = full_scale / 2^(bits -
# 1). This rounds every sample to the nearest step and clips anything outside
# the converter's range. The result stays in decimal numbers so later steps can
# use it. The default is 24-bit, like the proposal's recordings. It returns the
# rounded audio and a record with the bit depth, stage, step size, and how many
# samples had to be clipped. It stops with a ValueError if bits, full_scale, or
# the stage is bad, or if the audio isn't mono.
def quantize(audio, bits: int = 24, full_scale: float = 1.0, *, stage: str) -> tuple[np.ndarray, dict]:
    bits = _check_bits(bits)

    if full_scale <= 0:
        raise ValueError(f"full_scale must be positive, got {full_scale}")

    _check_stage(stage)
    samples = _mono(audio)

    # The size of one step, and the lowest and highest codes the converter can store.
    step = full_scale / 2 ** (bits - 1)
    lowest_code = -(2 ** (bits - 1))
    highest_code = 2 ** (bits - 1) - 1

    codes = np.round(samples / step)

    too_low = codes < lowest_code
    too_high = codes > highest_code
    clipped = int(np.count_nonzero(too_low | too_high))
    codes = np.clip(codes, lowest_code, highest_code)

    record = {
        "bits": bits,
        "stage": stage,
        "full_scale": float(full_scale),
        "step": step,
        "clipped_samples": clipped,
    }

    return codes * step, record
