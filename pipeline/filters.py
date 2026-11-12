"""
Builds the filters we apply on purpose, runs audio through them & keeps a record of exactly what was applied.

A filter spec is a small dict that describes a filter, for example:

    {"type": "highpass", "family": "butter", "order": 4, "cutoff": 250}
    {"type": "bandpass", "family": "butter", "order": 4, "cutoff": [250, 6000]}

Our proposal uses Butterworth filters at orders 1, 2 & 4. The other families are only here in case we add them later.

The order rule: "order" is always the total filter order, which is the number of poles. A band-pass is a high-pass at
the low edge followed by a low-pass at the high edge, the same way the C filters in pipeline/reversable_filters/ build
it, and each side gets half the poles. So a band-pass "order" has to be even, and order 8 means a 4th-order high-pass &
a 4th-order low-pass. Under this rule a band-pass can't be 1st order. Code that counts each side by itself, like a DAW,
should use half of our band-pass order. Don't call SciPy's own band-pass here: it builds a different shape, and it
doubles the order it's handed.

Cutoffs are in Hz. For "butter", the cutoff is the -3 dB point, which is where the filter has turned the level down by 3
dB. For "cheby1" & "ellip", it's where the passband ripple ends (-passband_ripple_db). For "cheby2", it's where the
stopband starts (-stopband_atten_db).

The filtering runs once over the whole clip, start to finish, from a silent starting state. That way the record
describes the filter that was really applied. Don't swap in forward-and-backward filtering or reset the filter between
chunks. Both change what the filter does.

This file has three groups:
    * Checking a filter spec
    * Designing and applying a filter
    * Measuring a designed filter
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy import signal

FILTER_TYPES = ("lowpass", "highpass", "bandpass")
FAMILIES = ("butter", "cheby1", "cheby2", "ellip")

# Defaults for the families that need them. Butterworth needs neither.
DEFAULT_PASSBAND_RIPPLE_DB = 1.0
DEFAULT_STOPBAND_ATTEN_DB = 40.0

# Every key a spec is allowed to have. Anything else gets treated as a typo.
_SPEC_KEYS = {"type", "family", "order", "cutoff", "passband_ripple_db", "stopband_atten_db"}

_FAMILIES_WITH_RIPPLE = ("cheby1", "ellip")
_FAMILIES_WITH_ATTENUATION = ("cheby2", "ellip")

# Saved in every record, so anyone reading it knows how the filter was run.
_PROCESSING_NOTE = "causal sosfilt, single pass, zero initial state"


# ********************************
# Checking a filter spec
# ********************************


# Stops if the spec has a key we don't know, like a typo'd "cutof".
def _check_keys(spec: dict) -> None:
    unknown = set(spec) - _SPEC_KEYS

    if unknown:
        raise ValueError(f"unknown filter spec keys: {sorted(unknown)}")


# Gives back the filter type, or stops if it isn't low-pass, high-pass, or band-pass.
def _check_type(spec: dict) -> str:
    kind = spec.get("type")

    if kind not in FILTER_TYPES:
        raise ValueError(f"filter type must be one of {FILTER_TYPES}, got {kind!r}")

    return kind


# Gives back the filter family, or stops if it's one we don't know. If the spec doesn't name a family, it's Butterworth.
def _check_family(spec: dict) -> str:
    family = spec.get("family", "butter")

    if family not in FAMILIES:
        raise ValueError(f"filter family must be one of {FAMILIES}, got {family!r}")

    return family


# Gives back the total order as a whole number. It stops if the order isn't a positive whole number, or if a band-pass
# order is odd.
def _check_order(spec: dict, kind: str) -> int:
    order = spec.get("order")

    # Python treats True as the number 1, so True and False are ruled out on purpose. 4.0 and "4" are ruled out too,
    # since they're probably typos.
    is_whole_number = isinstance(order, Integral) and not isinstance(order, bool)

    if not is_whole_number or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}")

    order = int(order)

    # A band-pass has a low edge and a high edge, and each edge uses the same number of poles, so the total is always
    # even.
    if kind == "bandpass" and order % 2 != 0:
        raise ValueError(f"band-pass order must be even (total poles), got {order}")

    return order


# Gives back a band-pass cutoff as [low, high] in Hz. It stops if the cutoff isn't a pair of 2 numbers, or if it doesn't
# fit 0 < low < high < Nyquist (half the sample rate).
def _check_bandpass_cutoff(cutoff, nyquist: float) -> list[float]:
    if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
        raise ValueError(f"band-pass cutoff must be [low, high] in Hz, got {cutoff!r}")

    low = float(cutoff[0])
    high = float(cutoff[1])

    if not 0 < low < high < nyquist:
        raise ValueError(f"band-pass cutoffs must satisfy 0 < low < high < {nyquist} Hz, got {cutoff!r}")

    return [low, high]


# Gives back a low-pass or high-pass cutoff in Hz. It stops if the cutoff isn't a number, or isn't between 0 & Nyquist
# (half the sample rate).
def _check_single_cutoff(cutoff, kind: str, nyquist: float) -> float:
    # True counts as a number in Python, so it's ruled out on purpose.
    is_number = isinstance(cutoff, Real) and not isinstance(cutoff, bool)

    if not is_number:
        # ValueError on purpose: every bad filter setting raises the same kind of error.
        raise ValueError(f"{kind} cutoff must be a number in Hz, got {cutoff!r}")

    if not 0 < cutoff < nyquist:
        raise ValueError(f"cutoff must be between 0 and {nyquist} Hz, got {cutoff}")

    return float(cutoff)


# Checks a filter spec & gives back a complete copy with every default filled in. The spec is a dict like the ones at
# the top of this file, and the sample rate sets the highest cutoff allowed (half of it). The copy always has "type",
# "family", "order" & "cutoff". Chebyshev & elliptic filters also get "passband_ripple_db" and/or "stopband_atten_db".
# It stops with a ValueError if any setting is missing, unknown, or out of range.
def normalize_spec(spec: dict, sample_rate: float) -> dict:
    _check_keys(spec)
    kind = _check_type(spec)
    family = _check_family(spec)
    order = _check_order(spec, kind)

    # Nyquist is the highest frequency the sample rate can hold.
    nyquist = sample_rate / 2

    if kind == "bandpass":
        cutoff_hz = _check_bandpass_cutoff(spec.get("cutoff"), nyquist)
    else:
        cutoff_hz = _check_single_cutoff(spec.get("cutoff"), kind, nyquist)

    full = {"type": kind, "family": family, "order": order, "cutoff": cutoff_hz}

    if family in _FAMILIES_WITH_RIPPLE:
        ripple = spec.get("passband_ripple_db", DEFAULT_PASSBAND_RIPPLE_DB)
        full["passband_ripple_db"] = float(ripple)

    if family in _FAMILIES_WITH_ATTENUATION:
        attenuation = spec.get("stopband_atten_db", DEFAULT_STOPBAND_ATTEN_DB)
        full["stopband_atten_db"] = float(attenuation)

    return full


# ********************************
# Designing and applying a filter
# ********************************


# Designs 1 high-pass or low-pass in the spec's family & gives it back as second-order sections (SOS), which is the form
# SciPy stores a filter in. kind is "highpass" or "lowpass", order is its number of poles & cutoff_hz is its 1 edge.
def _design_one_side(full_spec: dict, kind: str, order: int, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    # The ripple and attenuation are None for Butterworth, which is what SciPy expects.
    sos = signal.iirfilter(
        order,
        cutoff_hz,
        rp=full_spec.get("passband_ripple_db"),
        rs=full_spec.get("stopband_atten_db"),
        btype=kind,
        ftype=full_spec["family"],
        output="sos",
        fs=sample_rate,
    )

    return sos


# Designs a filter from a spec & gives it back as SciPy second-order sections: an array with 1 row of 6 numbers per
# section. The spec gets checked first, so it stops with a ValueError if the spec is bad.
def design_filter(spec: dict, sample_rate: float) -> np.ndarray:
    full = normalize_spec(spec, sample_rate)

    if full["type"] != "bandpass":
        return _design_one_side(full, full["type"], full["order"], full["cutoff"], sample_rate)

    # A band-pass is a high-pass at the low edge and a low-pass at the high edge, and each side gets half of the poles.
    low_hz, high_hz = full["cutoff"]
    poles_per_side = full["order"] // 2

    highpass = _design_one_side(full, "highpass", poles_per_side, low_hz, sample_rate)
    lowpass = _design_one_side(full, "lowpass", poles_per_side, high_hz, sample_rate)

    # Stacking the sections runs the high-pass first, then the low-pass.
    sos = np.vstack([highpass, lowpass])

    return sos


# Runs mono audio through a filter once, start to finish, from a silent starting state. It gives back the filtered audio
# & a record of what was applied: the full spec, the sample rate, how it was run & the filter's numbers ("sos"), so the
# exact filter can be rebuilt later. The record only holds plain numbers & lists, so it can be saved as JSON. It stops
# with a ValueError if the audio isn't mono or the spec is bad.
def apply_filter(audio, sample_rate: float, spec: dict) -> tuple[np.ndarray, dict]:
    samples = np.asarray(audio, dtype=np.float64)

    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {samples.shape}")

    full = normalize_spec(spec, sample_rate)
    sos = design_filter(full, sample_rate)
    filtered = signal.sosfilt(sos, samples)

    record = dict(full)
    record["sample_rate"] = float(sample_rate)
    record["processing"] = _PROCESSING_NOTE
    record["sos"] = sos.tolist()

    return filtered, record


# ********************************
# Measuring a designed filter
# ********************************


# Tells you what a designed filter does to each frequency: its gain in dB at 1 frequency or a list of them (in Hz). 0 dB
# means unchanged & -6 dB is about half the level. Very deep cuts bottom out near -6000 dB so the math doesn't break.
def response_db(sos: np.ndarray, sample_rate: float, freqs) -> np.ndarray:
    # Make sure we have a list of frequencies, even if we were given just one.
    freqs_hz = np.atleast_1d(np.asarray(freqs, dtype=np.float64))

    # Ask SciPy how the filter changes each frequency. The answer is complex numbers.
    _, response = signal.sosfreqz(sos, worN=freqs_hz, fs=sample_rate)

    # The size of each complex number is the gain. The tiny floor keeps log10 from failing on a gain of exactly 0.
    gain = np.maximum(np.abs(response), 1e-300)

    return 20 * np.log10(gain)
