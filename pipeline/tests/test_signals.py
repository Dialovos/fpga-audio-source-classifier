"""Checks for the simple test sounds.

The proposal's real data is 10-20 source recordings, with synthetic tones allowed as a supplement.
These sounds stand in until we have the recordings. Everything runs at 192 kHz, the proposal's sample rate.
"""

import numpy as np
import pytest

from pipeline.signals import thump, tone, white_noise

FS = 192_000


def peak_frequency(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    return np.fft.rfftfreq(len(audio), 1 / FS)[np.argmax(spectrum)]


# Tests: tone(100 Hz, 2 s) has exactly 2 seconds of samples, its strongest frequency is 100 Hz,
#   and its peak is the amplitude we asked for.
# Why: the tone is our stand-in for a periodic mechanical sound like the low-frequency rumble. If its pitch,
#   length, or level were off, any check that uses it would be measuring the wrong thing without us knowing.
def test_tone_has_the_right_length_pitch_and_level():
    audio = tone(100, 2.0, FS, amplitude=0.5)
    assert len(audio) == 2 * FS
    assert peak_frequency(audio) == pytest.approx(100, abs=0.5)
    assert np.max(np.abs(audio)) == pytest.approx(0.5, abs=1e-3)


# Tests: a thump is exactly zero before its onset, drops by more than 60 dB within half a second,
#   and never goes past the amplitude we asked for.
# Why: the thump is our stand-in for the proposal's short transients. Its onset time becomes the event's label,
#   so there can't be any sound before it, and it has to die away fast so it doesn't run into the next event.
def test_thump_is_silent_before_onset_and_dies_away():
    audio = thump(1.0, FS, onset_s=0.2, freq_hz=60, decay_s=0.05, amplitude=0.8)
    onset = int(0.2 * FS)
    assert np.all(audio[:onset] == 0)
    first_tenth = np.sqrt(np.mean(audio[onset : onset + FS // 10] ** 2))
    after_half_second = np.sqrt(np.mean(audio[onset + FS // 2 :] ** 2))
    assert after_half_second < first_tenth * 1e-3
    assert np.max(np.abs(audio)) <= 0.8


# Tests: white noise comes out at the exact RMS we asked for, the same seed gives the same noise,
#   a different seed gives different noise, and the average stays near zero.
# Why: noise is our stand-in for the broadband hiss and sets the noise floor, so its level has to be exact.
#   Repeating with the same seed is what lets us rebuild an example from its record. A non-zero average
#   would add a DC offset, which shows up as fake low-frequency energy.
def test_noise_hits_the_rms_and_repeats_with_the_same_seed():
    a = white_noise(1.0, FS, np.random.default_rng(7), rms=0.1)
    b = white_noise(1.0, FS, np.random.default_rng(7), rms=0.1)
    c = white_noise(1.0, FS, np.random.default_rng(8), rms=0.1)
    assert np.sqrt(np.mean(a**2)) == pytest.approx(0.1, rel=1e-12)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert abs(np.mean(a)) < 0.01


# Tests: the sound functions refuse a tone at Nyquist (96 kHz at 192 kHz), a zero-length clip, an amplitude
#   over 1, a thump onset past the end of the clip, zero decay, and a zero noise level.
# Why: each of these would give silence, an empty clip, a divide by zero, or audio that clips,
#   while the record still says the sound is there. That's a wrong label we wouldn't notice.
@pytest.mark.parametrize(
    "make",
    [
        lambda: tone(96_000, 1.0, FS),  # at Nyquist
        lambda: tone(100, 0, FS),
        lambda: tone(100, 1.0, FS, amplitude=1.5),
        lambda: thump(1.0, FS, onset_s=1.0),  # onset past the end
        lambda: thump(1.0, FS, decay_s=0),
        lambda: white_noise(1.0, FS, np.random.default_rng(1), rms=0),
    ],
)
def test_rejects_bad_settings(make):
    with pytest.raises(ValueError):
        make()


# Tests: white_noise raises a TypeError if it gets a plain number instead of a numpy Generator.
# Why: each example should use one generator made from its own seed. If functions took a bare seed and made
#   their own generator, two calls with the same seed would give the exact same noise, and that kind of repeat
#   is easy to miss.
def test_noise_needs_a_generator_not_a_bare_seed():
    with pytest.raises(TypeError):
        white_noise(1.0, FS, 7)
