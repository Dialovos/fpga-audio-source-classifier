"""
Spectrum features for the filter ID model.

A filter shows up as a change in the clip's average spectrum (how much energy it
has at each frequency). So the features are the clip's average power in 96 bands,
spaced evenly on a log scale, in dB. The median band is subtracted, so the overall
loudness doesn't matter, only the shape.

This file has two groups:
    * The band layout
    * Measuring the spectrum
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# The smallest power we take the log of. It stops silence from turning into -infinity.
SMALLEST_POWER = 1e-30


# ********************************
# The band layout
# ********************************


# Returns the edges of n_bands log-spaced bands from f_min_hz to f_max_hz. There's
# one more edge than bands, since each band sits between two edges.
def band_edges(n_bands: int, f_min_hz: float, f_max_hz: float) -> np.ndarray:
    return np.geomspace(f_min_hz, f_max_hz, n_bands + 1)


# Returns each band's center in Hz, which is the geometric middle of its two edges.
def band_centers(edges: np.ndarray) -> np.ndarray:
    lower_edges = edges[:-1]
    upper_edges = edges[1:]

    return np.sqrt(lower_edges * upper_edges)


# ********************************
# Measuring the spectrum
# ********************************


# Stops if the audio isn't mono, or if the band range doesn't fit below Nyquist.
def _check_inputs(samples: np.ndarray, sample_rate: float, f_min_hz: float, f_max_hz: float) -> None:
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {samples.shape}")

    nyquist = sample_rate / 2

    if not 0 < f_min_hz < f_max_hz <= nyquist:
        raise ValueError(f"need 0 < f_min_hz < f_max_hz <= {nyquist}, got {f_min_hz} and {f_max_hz}")


# Averages the power spectrum inside each band and returns one average per band.
# The spectrum comes in evenly spaced frequency "bins" (freqs from low to high, with
# the power in each). The lowest bands are narrower than one bin, so they have no
# bin inside them, and those bands use the power estimated at their center instead.
def _power_per_band(freqs: np.ndarray, power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bands = len(edges) - 1
    centers = band_centers(edges)

    # For each band, find the first bin at or above its lower edge,
    # and the first bin at or above its upper edge (the bin just past the band).
    first_bin = np.searchsorted(freqs, edges[:-1])
    end_bin = np.searchsorted(freqs, edges[1:])

    # A running total of the power: running_total[k] is the sum of the first k bins.
    # The sum of the bins in a band is then running_total[end] - running_total[first].
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


# Returns a clip's spectral shape: the average power in each band, in dB, minus
# the median band. The bands run from f_min_hz to f_max_hz, which has to be at or
# below Nyquist, and segment is how many samples Welch's method looks at at a time
# (it's shortened if the clip is shorter). A louder or quieter copy of the same
# clip gives the same numbers. It stops with a ValueError if the audio isn't mono
# or the band range is bad.
def spectrum_features(
    audio,
    sample_rate: float,
    n_bands: int = 96,
    f_min_hz: float = 20.0,
    f_max_hz: float = 24_000.0,
    segment: int = 32_768,
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
