"""
Layer 1, step 1: turn a clip into 96 numbers that describe the shape of its sound.

A filter doesn't change what's being said. It changes how loud each frequency range is. So we measure how loud the clip
is in 96 frequency bands & hand those 96 numbers (the "features") to the model.

The bands are spaced by ratio & not by a fixed number of Hz, so each band is ~8% higher than the one before it. That's
called log spacing, and it gives the low end as much detail as the high end. Each band's loudness is in dB. At the end
we subtract the middle (median) band from all of them, so a loud clip & a quiet copy of it give the same numbers. Only
the shape is left, and the shape is what a filter changes.

This file has two groups:
    * The band layout
    * Measuring the spectrum
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# The smallest power we'll take the log of. Without it, pure silence would turn into minus infinity & break the math.
SMALLEST_POWER = 1e-30


# ********************************
# The band layout
# ********************************


# Gives back where each band starts & ends, in Hz, from f_min_hz up to f_max_hz. The edges are spaced by ratio (log
# spacing). There's always 1 more edge than there are bands, since every band sits between 2 edges.
def band_edges(n_bands: int, f_min_hz: float, f_max_hz: float) -> np.ndarray:
    return np.geomspace(f_min_hz, f_max_hz, n_bands + 1)


# Gives back the middle of each band in Hz. Since the bands are spaced by ratio, the middle is the geometric one: the
# square root of (lower edge x upper edge).
def band_centers(edges: np.ndarray) -> np.ndarray:
    lower_edges = edges[:-1]
    upper_edges = edges[1:]

    return np.sqrt(lower_edges * upper_edges)


# ********************************
# Measuring the spectrum
# ********************************


# Makes sure the inputs make sense before we measure anything. The audio has to be mono (1 channel), and the band range
# has to fit under Nyquist, which is half the sample rate & the highest frequency the audio can hold. If not, it stops
# with a ValueError that says what's wrong.
def _check_inputs(samples: np.ndarray, sample_rate: float, f_min_hz: float, f_max_hz: float) -> None:
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {samples.shape}")

    nyquist = sample_rate / 2

    if not 0 < f_min_hz < f_max_hz <= nyquist:
        raise ValueError(f"need 0 < f_min_hz < f_max_hz <= {nyquist}, got {f_min_hz} and {f_max_hz}")


# Gives back the average power inside each band. The spectrum we get from SciPy comes as evenly spaced frequency points
# called bins (freqs is where each bin is & power is how strong it is). For each band we average the bins that land
# inside it. The lowest bands are narrower than a single bin, so nothing lands inside them. For those we estimate the
# power at the band's center from the bins on either side.
def _power_per_band(freqs: np.ndarray, power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bands = len(edges) - 1
    centers = band_centers(edges)

    # For each band, find the first bin at or above its lower edge, and the first bin at or above its upper edge (the
    # bin just past the band).
    first_bin = np.searchsorted(freqs, edges[:-1])
    end_bin = np.searchsorted(freqs, edges[1:])

    # A running total of the power: running_total[k] is the sum of the first k bins. The sum of the bins in a band is
    # then running_total[end] - running_total[first].
    running_total = np.concatenate([[0.0], np.cumsum(power)])

    band_power = np.zeros(n_bands)

    for band in range(n_bands):
        first = first_bin[band]
        end = end_bin[band]
        bins_in_band = end - first

        if bins_in_band > 0:
            band_sum = running_total[end] - running_total[first]
            band_power[band] = band_sum / bins_in_band
        else:
            # No bins inside this band: estimate the power at its center from the bins around it.
            band_power[band] = np.interp(centers[band], freqs, power)

    return band_power


# This is the function everything else calls. It takes a clip & gives back its 96 numbers: the loudness of each band in
# dB, with the middle (median) band subtracted. The bands go from f_min_hz to f_max_hz. segment is how many samples get
# measured at a time (it gets shortened if the clip is shorter than that). A louder or quieter copy of the same clip
# gives the same numbers. It stops with a ValueError if the audio isn't mono or the band range is bad.
def spectrum_features(
    audio,
    sample_rate: float,
    n_bands: int = 96,
    f_min_hz: float = 20.0,
    f_max_hz: float = 24_000.0,
    segment: int = 8_192,
) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float64)
    _check_inputs(samples, sample_rate, f_min_hz, f_max_hz)

    # Welch's method: cut the clip into overlapping pieces, measure the power at each
    # frequency in every piece, and average them. That's steadier than one big measurement.
    segment_length = min(segment, len(samples))
    freqs, power = signal.welch(samples, fs=sample_rate, nperseg=segment_length)

    edges = band_edges(n_bands, f_min_hz, f_max_hz)
    band_power = _power_per_band(freqs, power, edges)

    db = 10 * np.log10(np.maximum(band_power, SMALLEST_POWER))

    # Subtract the median band, so only the shape is left, not the loudness.
    return db - np.median(db)
