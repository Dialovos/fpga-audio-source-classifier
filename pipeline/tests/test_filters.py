"""Checks that the filters we apply are the filters we asked for.

Everything runs at 192 kHz, the proposal's sample rate. The proposal uses Butterworth
high-pass, low-pass, and band-pass filters at orders 1, 2, and 4, so those are the main cases.
"""

import json

import numpy as np
import pytest

from pipeline.filters import apply_filter, design_filter, normalize_spec, response_db

FS = 192_000


def tone(freq_hz: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return np.sin(2 * np.pi * freq_hz * t)


def applied_gain_db(spec: dict, freq_hz: float) -> float:
    """Gain of the filter we actually applied, measured on a steady tone.

    Uses the last second only: it's past the start-up transient, and integer-Hz
    tones fit a whole number of cycles in it, so the RMS ratio is exact.
    """
    x = tone(freq_hz)
    y, _ = apply_filter(x, FS, spec)
    tail = slice(-FS, None)
    return 20 * np.log10(np.sqrt(np.mean(y[tail] ** 2)) / np.sqrt(np.mean(x[tail] ** 2)))


def prewarp(freq_hz: float) -> float:
    """Analog frequency that the digital (bilinear) filter maps to freq_hz."""
    return FS / np.pi * np.tan(np.pi * freq_hz / FS)


def butterworth_db(freq_hz, kind, prototype_order, cutoff):
    """Textbook analog Butterworth magnitude in dB, evaluated with prewarping."""
    f = prewarp(freq_hz)
    if kind == "lowpass":
        x = f / prewarp(cutoff)
    elif kind == "highpass":
        x = prewarp(cutoff) / f
    else:
        low, high = prewarp(cutoff[0]), prewarp(cutoff[1])
        x = (f**2 - low * high) / (f * (high - low))
    return -10 * np.log10(1 + x ** (2 * prototype_order))


# Tests: a Butterworth high-pass at 250 Hz cuts a 100 Hz tone by the textbook amount at orders 1, 2, and 4
#   (~8.6, 16.0, and 31.8 dB). It also checks that the textbook helper itself gives those numbers,
#   so we know the reference is right too.
# Why: the rumble in the proposal sits below the 250 Hz edge, and orders 1, 2, and 4 are the ones the proposal
#   starts with. If the filter we apply doesn't cut the rumble by the amount the label says, every example
#   built on it has a wrong label, and the model would learn from it anyway.
@pytest.mark.parametrize("order, textbook_db", [(1, -8.6), (2, -16.0), (4, -31.8)])
def test_highpass_cuts_100hz_rumble_by_textbook_amount(order, textbook_db):
    spec = {"type": "highpass", "family": "butter", "order": order, "cutoff": 250}
    expected = butterworth_db(100, "highpass", order, 250)
    assert expected == pytest.approx(textbook_db, abs=0.1)
    assert applied_gain_db(spec, 100) == pytest.approx(expected, abs=0.05)


# Tests: a Butterworth low-pass at 6 kHz cuts a 10 kHz tone by the textbook amount at orders 1, 2, and 4.
# Why: the hiss in the proposal sits above the 6 kHz edge. This is the same check as the rumble test,
#   but for the low-pass side and the top of the band.
@pytest.mark.parametrize("order", [1, 2, 4])
def test_lowpass_cuts_10khz_hiss_by_textbook_amount(order):
    spec = {"type": "lowpass", "family": "butter", "order": order, "cutoff": 6000}
    expected = butterworth_db(10_000, "lowpass", order, 6000)
    assert applied_gain_db(spec, 10_000) == pytest.approx(expected, abs=0.05)


# Tests: a 4th-order high-pass at 250 Hz leaves a 2 kHz tone at ~0 dB.
# Why: the tests above only check what the filter removes. A filter that cuts the right spot but also
#   turns down the speech band would still be the wrong filter, so this checks the other half of the response.
def test_highpass_passes_speech_band():
    spec = {"type": "highpass", "family": "butter", "order": 4, "cutoff": 250}
    assert applied_gain_db(spec, 2000) == pytest.approx(0.0, abs=0.05)


# Tests: a 250 Hz - 6 kHz Butterworth band-pass matches the textbook gain at 100 Hz (rumble), 1 kHz (speech),
#   and 10 kHz (hiss), at total orders 2, 4, and 8.
# Why: 250 Hz - 6 kHz is the band in the proposal's figure, and those three tones are the sounds it buries or
#   keeps. The proposal doesn't say how band-pass order is counted, so this covers both readings. If we count
#   total poles, its 2nd and 4th order are total orders 2 and 4 (1st order isn't possible). If we count each
#   side like a DAW, its 1st, 2nd, and 4th order are total orders 2, 4, and 8. It also checks our order rule,
#   since handing SciPy the wrong number breaks the match.
@pytest.mark.parametrize("total_order", [2, 4, 8])
@pytest.mark.parametrize("freq_hz", [100, 1000, 10_000])
def test_bandpass_matches_textbook(total_order, freq_hz):
    spec = {"type": "bandpass", "family": "butter", "order": total_order, "cutoff": [250, 6000]}
    expected = butterworth_db(freq_hz, "bandpass", total_order // 2, [250, 6000])
    assert applied_gain_db(spec, freq_hz) == pytest.approx(expected, abs=0.05)


# Tests: the gain we measure on a real tone after filtering matches the gain of the designed filter at 100 Hz,
#   1 kHz, and 10 kHz. It covers a Butterworth filter of each type, plus one Chebyshev I, one Chebyshev II,
#   and one elliptic filter.
# Why: response_db only looks at the design. This checks that apply_filter really runs that design on the
#   audio (same coefficients, one pass, zero start), so the label matches the audio and not just the plan.
#   The other three families aren't in the proposal, but filters.py supports them, so they get checked too.
@pytest.mark.parametrize(
    "spec",
    [
        {"type": "highpass", "family": "butter", "order": 1, "cutoff": 250},
        {"type": "lowpass", "family": "butter", "order": 2, "cutoff": 6000},
        {"type": "bandpass", "family": "butter", "order": 4, "cutoff": [250, 6000]},
        {"type": "lowpass", "family": "cheby1", "order": 6, "cutoff": 3000},
        {"type": "lowpass", "family": "cheby2", "order": 5, "cutoff": 2000},
        {"type": "bandpass", "family": "ellip", "order": 6, "cutoff": [300, 3400]},
    ],
)
@pytest.mark.parametrize("freq_hz", [100, 1000, 10_000])
def test_applied_filter_matches_designed_response(spec, freq_hz):
    designed = response_db(design_filter(spec, FS), FS, freq_hz)[0]
    assert applied_gain_db(spec, freq_hz) == pytest.approx(designed, abs=0.05)


# Tests: at the cutoff frequency, each filter's gain is what its definition says: -3.01 dB for Butterworth
#   (at orders 1, 2, and 4), minus the passband ripple for Chebyshev I and elliptic, and minus the stopband
#   attenuation for Chebyshev II.
# Why: the AI has to predict cutoffs, and we'll score it on cutoff error, so the cutoff in a label has to mean
#   one exact point. For Butterworth that's the -3 dB point. The other families define it differently,
#   so they're checked too in case we ever add them.
@pytest.mark.parametrize(
    "spec, expected_db",
    [
        ({"type": "lowpass", "family": "butter", "order": 1, "cutoff": 6000}, -3.01),
        ({"type": "highpass", "family": "butter", "order": 2, "cutoff": 250}, -3.01),
        ({"type": "lowpass", "family": "butter", "order": 4, "cutoff": 6000}, -3.01),
        ({"type": "lowpass", "family": "cheby1", "order": 4, "cutoff": 3000, "passband_ripple_db": 0.5}, -0.5),
        ({"type": "lowpass", "family": "cheby2", "order": 4, "cutoff": 3000, "stopband_atten_db": 50}, -50.0),
        ({"type": "lowpass", "family": "ellip", "order": 4, "cutoff": 3000}, -1.0),
    ],
)
def test_cutoff_lands_where_we_asked(spec, expected_db):
    sos = design_filter(spec, FS)
    assert response_db(sos, FS, spec["cutoff"])[0] == pytest.approx(expected_db, abs=0.05)


# Tests: both edges of a 250 Hz - 6 kHz Butterworth band-pass sit at -3.01 dB, at total orders 2, 4, and 8.
# Why: band-pass cutoffs go in as a pair, so it's easy to swap them or scale them wrong. The AI predicts both
#   edges, so this checks that both land where the label says they are.
@pytest.mark.parametrize("total_order", [2, 4, 8])
def test_bandpass_edges_land_where_we_asked(total_order):
    spec = {"type": "bandpass", "family": "butter", "order": total_order, "cutoff": [250, 6000]}
    edges = response_db(design_filter(spec, FS), FS, [250, 6000])
    assert edges == pytest.approx([-3.01, -3.01], abs=0.05)


# Tests: the number of second-order sections follows our order rule. Each section holds up to 2 poles,
#   so high-pass and low-pass orders 1 and 2 give 1 section and order 4 gives 2. Band-pass orders 2, 4,
#   and 8 give 1, 2, and 4 sections.
# Why: the AI has to predict the order, so the order in a label has to mean one thing. SciPy's band-pass with
#   order N really ends up with order 2N, and our labels count total poles, so a band-pass labeled 4 has to be
#   4 poles (2 sections), not 8. Keeping this rule in one place is the main reason we have our own filter function.
@pytest.mark.parametrize(
    "kind, order, sections",
    [
        ("lowpass", 1, 1),
        ("lowpass", 2, 1),
        ("lowpass", 4, 2),
        ("highpass", 1, 1),
        ("highpass", 2, 1),
        ("highpass", 4, 2),
        ("bandpass", 2, 1),
        ("bandpass", 4, 2),
        ("bandpass", 8, 4),
    ],
)
def test_order_counts_total_poles(kind, order, sections):
    cutoff = [250, 6000] if kind == "bandpass" else 1000
    sos = design_filter({"type": kind, "family": "butter", "order": order, "cutoff": cutoff}, FS)
    assert sos.shape == (sections, 6)


# Tests: the record from apply_filter survives a trip through JSON, has every setting we used (defaults
#   included), and redesigning the filter from those saved settings gives the same coefficients that were saved.
#   It runs on the proposal's Butterworth band-pass and on an elliptic filter, which has the most defaults.
# Why: the proposal keeps the exact filter settings for every example as ground truth, and we'll also run
#   compensation with the known filter to compare against the AI's estimates. Both need a record that rebuilds
#   the exact filter. If a default weren't saved, changing that default later would quietly change old labels.
@pytest.mark.parametrize(
    "spec",
    [
        {"type": "bandpass", "family": "butter", "order": 4, "cutoff": [250, 6000]},
        {"type": "bandpass", "family": "ellip", "order": 6, "cutoff": [300, 3400]},
    ],
)
def test_record_is_json_and_rebuilds_the_same_filter(spec):
    _, record = apply_filter(tone(1000, seconds=0.1), FS, spec)
    restored = json.loads(json.dumps(record))
    full = normalize_spec(spec, FS)
    assert {key: restored[key] for key in full} == full
    assert restored["sample_rate"] == FS
    np.testing.assert_allclose(design_filter(full, restored["sample_rate"]), restored["sos"])


# Tests: design_filter refuses settings that don't make sense: an odd band-pass order, reversed cutoffs,
#   a cutoff at Nyquist (96 kHz at 192 kHz), order 0, a float order, an unknown type or family, and a typo'd key.
# Why: a bad setting that gets quietly fixed or ignored makes a label that doesn't match the audio, and a wrong
#   label is hard to spot once it's in the dataset. We'd rather the generator stop than save a wrong example.
#   The odd band-pass case follows our total-pole rule, so it changes if the team picks the DAW way of counting.
@pytest.mark.parametrize(
    "spec",
    [
        {"type": "bandpass", "order": 3, "cutoff": [250, 6000]},  # odd band-pass order
        {"type": "bandpass", "order": 4, "cutoff": [6000, 250]},  # cutoffs reversed
        {"type": "lowpass", "order": 4, "cutoff": 96_000},  # at Nyquist
        {"type": "lowpass", "order": 0, "cutoff": 3000},
        {"type": "lowpass", "order": 4.0, "cutoff": 3000},
        {"type": "notch", "order": 4, "cutoff": 3000},
        {"type": "lowpass", "family": "bessel", "order": 4, "cutoff": 3000},
        {"type": "lowpass", "order": 4, "cutoff": 3000, "cutof": 2000},  # typo'd key
    ],
)
def test_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        design_filter(spec, FS)


# Tests: apply_filter refuses 2-channel audio.
# Why: the pipeline works on one channel at a time. SciPy would filter a stereo array along its last axis
#   (across the 2 channels instead of across time) without any error, so we have to catch it ourselves.
def test_rejects_stereo_audio():
    with pytest.raises(ValueError):
        apply_filter(np.zeros((100, 2)), FS, {"type": "lowpass", "order": 2, "cutoff": 3000})
