"""
Checks for the filter ID model's spectrum features.
"""

import numpy as np
from scipy import signal

from pipeline.filter_id.features import band_centers, band_edges, spectrum_features

FS = 192_000


# Tests: the band layout has one more edge than bands, starts and ends where we asked, gives every band the same
#   width in octaves, and puts each center at the geometric middle of its edges.
# Why: every feature is the power in one of these bands. If the edges were off or uneven, the same filter would
#   show up differently depending on where its cutoff fell, and the model would learn the band layout instead.
def test_bands_are_evenly_spaced_on_a_log_scale():
    edges = band_edges(96, 20.0, 24_000.0)
    assert len(edges) == 97
    assert edges[0] == 20.0
    assert np.isclose(edges[-1], 24_000.0)
    np.testing.assert_allclose(np.diff(np.log2(edges)), np.log2(24_000 / 20) / 96)
    np.testing.assert_allclose(band_centers(edges), np.sqrt(edges[:-1] * edges[1:]))


# Tests: the features have one value per band, don't change when the clip gets louder, and show a 4th-order
#   high-pass at 1 kHz as low bands sitting more than 30 dB below a flat passband.
# Why: the model should learn the filter's shape, not how loud the clip was. And if a clear high-pass didn't show
#   up as a low-end drop in the features, no model could find it from them. The features are relative to their
#   median band, so the check compares the low bands to the passband, not to zero.
def test_features_ignore_level_and_show_the_filter():
    noise = np.random.default_rng(2).standard_normal(FS)
    flat = spectrum_features(noise, FS)
    assert flat.shape == (96,)
    np.testing.assert_allclose(spectrum_features(10 * noise, FS), flat, atol=1e-9)
    highpass = signal.butter(4, 1000, btype="highpass", fs=FS, output="sos")
    filtered = signal.sosfilt(highpass, noise)
    change = spectrum_features(filtered, FS) - flat
    centers = band_centers(band_edges(96, 20, 24_000))
    passband = change[centers > 4000]
    assert np.ptp(passband) < 3
    assert np.all(change[centers < 250] - np.median(passband) < -30)
