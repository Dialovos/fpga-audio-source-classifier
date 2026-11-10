"""
Check the Python inverse against hand-built blocks and the C wrapper against a small script.

This file has three groups:
    * Hand-built inverse
    * Recovery checks
    * C wrapper checks
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from scipy import signal

from pipeline.compensation import c_inverse, compensate, python_inverse
from pipeline.filters import apply_filter, design_filter

# ********************************
# Hand-built inverse
# ********************************

FS = 48000
SPEC = {
    "type": "lowpass",
    "family": "butter",
    "order": 4,
    "cutoff": 2000,
}
N_FFT = 65536
GUARD = 2048
HOP = N_FFT - 2 * GUARD


# Returns a hand-built inverse block at one offset with the plan's constants.
def hand_block(audio, offset):
    block = np.zeros(N_FFT)

    for position in range(N_FFT):
        source = offset - GUARD + position

        if 0 <= source < len(audio):
            block[position] = audio[source]

    angles = 2 * np.pi * np.arange(N_FFT // 2 + 1) / N_FFT
    _, response = signal.sosfreqz(design_filter(SPEC, FS), worN=angles)
    inverse = np.conj(response) / (np.abs(response)**2 + 0.001)
    recovered = np.fft.irfft(np.fft.rfft(block) * inverse, N_FFT)

    return recovered[GUARD:GUARD + min(HOP, len(audio) - offset)]


# Tests: short audio and more than two hops match independently built blocks to
# 1e-12.
# Why: exact padding and keep positions matter at the first, last and joined
# blocks.
def test_inverse_matches_hand_built_blocks():
    for count in (10003, 2 * HOP + 591):
        audio = np.random.default_rng(3).standard_normal(count)
        output = python_inverse(audio, FS, SPEC)

        assert output.dtype == np.float64
        assert len(output) == count

        for offset in range(0, count, HOP):
            expected = hand_block(audio, offset)
            np.testing.assert_allclose(output[offset:offset + len(expected)], expected, atol=1e-12, rtol=0)


# ********************************
# Recovery checks
# ********************************


# Tests: Welch power after recovery meets the passband and under-20-dB loss
# limits.
# Why: the copied inverse must recover useful frequencies, not just follow the
# same indexing.
def test_inverse_recovers_the_useful_spectrum():
    audio = np.random.default_rng(4).standard_normal(FS * 2)
    filtered, _ = apply_filter(audio, FS, SPEC)
    recovered = python_inverse(filtered, FS, SPEC)
    frequencies, original = signal.welch(audio[4096:-4096], fs=FS, nperseg=16384)
    _, restored = signal.welch(recovered[4096:-4096], fs=FS, nperseg=16384)
    _, response = signal.sosfreqz(design_filter(SPEC, FS), worN=frequencies, fs=FS)
    loss = -20 * np.log10(np.maximum(np.abs(response), 1e-30))
    error = abs(10 * np.log10(restored / original))
    passband = (frequencies >= 100) & (loss < 3)
    useful = (frequencies >= 100) & (loss < 20)

    assert np.max(error[passband]) < 0.5
    assert np.max(error[useful]) < 1


# Tests: an impulse at a block join agrees with the shifted middle impulse
# within 1% of its peak.
# Why: a block join must not add a false transient to the event comparison.
def test_inverse_block_join_error():
    count = 3 * HOP
    join = np.zeros(count)
    middle = np.zeros(count)
    join[HOP] = 1
    middle[HOP + HOP // 2] = 1
    joined = python_inverse(join, FS, SPEC)
    centered = python_inverse(middle, FS, SPEC)
    radius = 4096
    actual = joined[HOP - radius:HOP + radius]
    expected = centered[HOP + HOP // 2 - radius:HOP + HOP // 2 + radius]
    difference = float(np.max(abs(actual - expected)))
    fraction = difference / float(np.max(abs(expected)))
    print(f"block_join_error={difference:.12g}, peak_fraction={fraction:.12g}, percent={100 * fraction:.12g}")

    assert fraction < 0.01


# Tests: peaks over full scale survive and invalid specs and methods stop.
# Why: FLOAT recovery must keep loud peaks without silently accepting
# unsupported C settings.
def test_inverse_does_not_clip_and_checks_settings():
    audio = np.full(10000, 2.0)
    recovered, record = compensate(audio, FS, SPEC)

    assert len(recovered) == len(audio)
    assert recovered.max() > 1
    assert record["peak"] == np.max(abs(recovered))
    assert record["lam"] == 0.001

    for changed in ({"family": "ellip"}, {"order": 3}, {"sos": []}):
        with pytest.raises(ValueError):
            compensate(audio, FS, {**SPEC, **changed})

    with pytest.raises(ValueError, match="method"):
        compensate(audio, FS, SPEC, method="wrong")


# ********************************
# C wrapper checks
# ********************************


# Returns a fake C command and its log path after writing the script. The log
# stays outside the program's temporary folder so it can be checked afterward.
def fake_filter(tmp_path, mode="copy"):
    script = tmp_path / f"fake_{mode}.py"
    log = tmp_path / f"arguments_{mode}.json"
    script.write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import soundfile as sf\n"
        f"Path({str(log)!r}).write_text(json.dumps({{'args': sys.argv[1:], 'cwd': os.getcwd()}}))\n"
        "audio, rate = sf.read('in.wav')\n"
        f"mode = {mode!r}\n"
        "if mode == 'short':\n"
        "    audio = audio[:-1]\n"
        "if mode != 'missing':\n"
        "    sf.write('out.wav', audio, rate, subtype='FLOAT')\n"
        "print('fake filter output')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )

    return [sys.executable, str(script)], log


# Tests: all filter arguments, half order, exit 1, FLOAT precision and separate
# folders are right.
# Why: the real binary's unusual exit code and shared int.wav need this exact
# wrapper.
def test_c_arguments_and_private_folders(tmp_path):
    command, log = fake_filter(tmp_path)
    audio = np.array([0.0, 0.5, 2.0, -2.0])
    highpass = {
        "type": "highpass",
        "family": "butter",
        "order": 2,
        "cutoff": 250,
    }
    bandpass = {
        "type": "bandpass",
        "family": "butter",
        "order": 8,
        "cutoff": [250, 6000],
    }
    specs = [SPEC, highpass, bandpass]

    expected = [
        ["LP", "butter", "4", "0", "2000.0", "in.wav", "out.wav", "4"],
        ["HP", "butter", "2", "250.0", "0", "in.wav", "out.wav", "5"],
        [
            "BP",
            "butter",
            "4",
            "250.0",
            "6000.0",
            "in.wav",
            "out.wav",
            "6",
        ],
    ]
    folders = set()

    for spec, args in zip(specs, expected):
        output = c_inverse(audio, FS, spec, command)
        np.testing.assert_array_equal(output, audio)
        saved = json.loads(log.read_text())

        assert saved["args"] == args
        assert saved["cwd"] != str(tmp_path)

        folders.add(saved["cwd"])

    assert len(folders) == 3


# Tests: missing or wrong-length out.wav stops and includes the program's
# output.
# Why: exit 1 alone does not prove that the inverse made a valid output.
def test_c_bad_outputs_stop(tmp_path):
    for mode in ("short", "missing"):
        command, _ = fake_filter(tmp_path, mode)

        with pytest.raises(ValueError, match="fake filter output"):
            c_inverse(np.ones(100), FS, SPEC, command)
