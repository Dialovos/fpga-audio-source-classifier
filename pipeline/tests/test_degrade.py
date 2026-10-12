"""Checks for added noise and bit-depth rounding.

Everything runs at 192 kHz, and rounding defaults to 24-bit, like the proposal's source recordings.
"""

import numpy as np
import pytest

from pipeline.degrade import add_noise, quantize

FS = 192_000


def sine(freq_hz: float, amplitude: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


# Tests: add_noise hits the SNR we ask for (0, 10, and 30 dB) almost exactly, and its record holds that SNR
#   and the noise level it really added.
# Why: the proposal adds noise at several SNRs, and heavily cut sounds may not come back at practical noise
#   levels. Our results depend on the SNR being exactly what the record says.
@pytest.mark.parametrize("snr_db", [0.0, 10.0, 30.0])
def test_noise_hits_the_requested_snr(snr_db):
    clean = sine(440, 0.5)
    noisy, record = add_noise(clean, snr_db, np.random.default_rng(3), stage="after_filter")
    noise = noisy - clean
    measured = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    assert measured == pytest.approx(snr_db, abs=1e-9)
    assert record["snr_db"] == snr_db
    assert record["noise_rms"] == pytest.approx(np.sqrt(np.mean(noise**2)))


# Tests: add_noise saves whether the noise went in before or after the filter, and refuses any other stage.
# Why: the proposal tests noise added before the filter against noise added after it. Those give different
#   audio for the same SNR, so the record has to say which one happened, or we can't compare the two.
@pytest.mark.parametrize("stage", ["before_filter", "after_filter"])
def test_noise_records_whether_it_went_before_or_after_the_filter(stage):
    _, record = add_noise(sine(440, 0.5), 20, np.random.default_rng(5), stage=stage)
    assert record["stage"] == stage
    with pytest.raises(ValueError):
        add_noise(sine(440, 0.5), 20, np.random.default_rng(5), stage="middle")


# Tests: the same seed gives the same noisy audio.
# Why: so we can rebuild any example from its record and get the same audio back.
def test_noise_repeats_with_the_same_seed():
    clean = sine(440, 0.5)
    a, _ = add_noise(clean, 20, np.random.default_rng(11), stage="before_filter")
    b, _ = add_noise(clean, 20, np.random.default_rng(11), stage="before_filter")
    np.testing.assert_array_equal(a, b)


# Tests: add_noise refuses silent audio (ValueError) and a bare seed instead of a numpy Generator (TypeError).
# Why: an SNR on silence is a divide by zero, so there's no right answer to give. The bare seed check is for
#   the same reason as in white_noise: one generator per example, so noise doesn't repeat by accident.
def test_noise_rejects_silence_and_bare_seeds():
    with pytest.raises(ValueError):
        add_noise(np.zeros(100), 20, np.random.default_rng(1), stage="after_filter")
    with pytest.raises(TypeError):
        add_noise(sine(440, 0.5), 20, 1, stage="after_filter")


# Tests: quantize defaults to 24 bits, every sample lands on the 24-bit grid (step = 2^-23), no sample moved
#   more than half a step, nothing clipped, and the record saves the stage.
# Why: the proposal's source recordings are 24-bit, so that's the default when we round. If samples landed off
#   the grid or moved more than half a step, we'd be adding a different kind of damage than the record says.
def test_quantize_defaults_to_24_bits_and_lands_on_the_grid():
    audio = sine(997, 0.9)
    out, record = quantize(audio, stage="after_filter")
    step = record["step"]
    assert record["bits"] == 24
    assert step == 2**-23
    assert record["stage"] == "after_filter"
    np.testing.assert_allclose(out / step, np.round(out / step), atol=1e-9)
    assert np.max(np.abs(out - audio)) <= step / 2 + 1e-15
    assert record["clipped_samples"] == 0


# Tests: a full-scale sine rounded to 24 and 16 bits has an SNR of about 6.02 x bits + 1.76 dB
#   (146.2 and 98.1 dB), within 1 dB.
# Why: that's the standard formula for rounding noise, so it checks our rounding acts like a real converter.
#   24-bit is the proposal's source format, and 16-bit is there to compare against. It also tells us how much
#   each bit depth raises the noise floor, which affects what we can recover.
@pytest.mark.parametrize("bits, expected_db", [(24, 146.2), (16, 98.1)])
def test_full_scale_sine_gets_the_textbook_quantization_noise(bits, expected_db):
    audio = sine(997, 1 - 2**-(bits - 1))
    out, _ = quantize(audio, bits=bits, stage="after_filter")
    snr = 10 * np.log10(np.mean(audio**2) / np.mean((out - audio) ** 2))
    assert snr == pytest.approx(expected_db, abs=1.0)


# Tests: values past full scale get clipped to the converter's range (-1 up to one step below 1),
#   and the record counts how many samples clipped. 1.0 counts as clipped because a signed converter's
#   top value is one step below full scale.
# Why: a real converter clips, and clipping adds distortion that has nothing to do with the filter.
#   The count lets us find or drop clipped examples instead of training on them without knowing.
def test_quantize_clips_and_counts_overloads():
    out, record = quantize(np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]), bits=24, stage="after_filter")
    assert out.max() == pytest.approx(1 - 2**-23)
    assert out.min() == -1.0
    assert record["clipped_samples"] == 3  # -2.0, 1.0, and 2.0 are outside [-1, 1 - step]


# Tests: quantize refuses 1 bit, 33 bits, a float like 24.0, True, and an unknown stage.
# Why: we only allow 2 to 32 bits. 1 bit leaves just two levels (-1 and 0), and nothing we'd record with goes
#   past 32. True counts as 1 in Python and 24.0 looks fine but isn't a whole number, so both could slip in
#   from a typo in a settings file. A wrong stage would make the record lie about where the rounding happened.
@pytest.mark.parametrize(
    "settings",
    [
        {"bits": 1, "stage": "after_filter"},
        {"bits": 33, "stage": "after_filter"},
        {"bits": 24.0, "stage": "after_filter"},
        {"bits": True, "stage": "after_filter"},
        {"bits": 24, "stage": "middle"},
    ],
)
def test_quantize_rejects_bad_settings(settings):
    with pytest.raises(ValueError):
        quantize(sine(440, 0.5), **settings)
