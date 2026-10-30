"""
Checks for the C filter program in pipeline/reversable_filters/.

These build the C program and run it the way the rest of the project will, as a program on 32-bit float WAV
files. Options 1-3 (applying a filter) get compared with SciPy's Butterworth filters, and options 4-6 (undoing a
filter) with the same regularized inverse written in Python. The C code itself isn't changed.

They need Linux with gcc, make, and FFTW (on Ubuntu: sudo apt install build-essential libfftw3-dev), so the file
isn't named test_*.py and the normal test run leaves it out. Run it on its own from the repo root:

    python -m pytest pipeline/tests/check_c_filter.py
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

C_FOLDER = Path(__file__).parents[1] / "reversable_filters"
FS = 192_000

# The same settings the C program uses to undo a filter: the regularization (lambda), the FFT size, and how many
# samples get thrown away at each end of every block.
LAMBDA = 0.001
N_FFT = 65_536
GUARD = 2048

# The cutoffs from the example commands in the C folder's readme.
HIGHPASS_HZ = 300.0
LOWPASS_HZ = 5000.0

# The C program counts the order on each side of a band-pass, and only takes these.
ORDERS = [1, 2, 4]

# How close the C output has to be to Python, as a share of the signal's peak. The C program saves 32-bit floats,
# which round at about 6e-8 of the peak, so this leaves plenty of room for that and nothing for a wrong filter.
TOLERANCE = 1e-5


# Builds the C program once for this file and returns its path. The source files get copied to a temporary folder
# first, so the build never leaves files in the repo. The flags are the Makefile's, plus one: newer gcc (14 and up)
# stops on a call to a function that wasn't declared, and main.c has two right now (strcmp without <string.h>, and
# reverse_bilinear_highpass, which isn't in Butterworth.h). The extra flag turns those back into warnings. If the
# build fails, every check fails with the compiler's output.
@pytest.fixture(scope="module")
def program(tmp_path_factory):
    if shutil.which("make") is None or shutil.which("gcc") is None:
        pytest.fail("gcc or make isn't installed. These checks need Linux with gcc, make, and FFTW.")

    folder = tmp_path_factory.mktemp("c_filter")
    for name in ["main.c", "Butterworth.c", "Butterworth.h", "Makefile"]:
        shutil.copy(C_FOLDER / name, folder / name)

    flags = "CFLAGS=-Wall -Wextra -O2 -std=c11 -Wno-error=implicit-function-declaration"
    result = subprocess.run(["make", flags], cwd=folder, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(f"the C program didn't build:\n{result.stdout}{result.stderr}")

    return folder / "filter"


# Returns 1 s of white noise at an RMS of 0.1, already rounded to 32-bit floats. The C program reads 32-bit floats,
# so rounding first means the C program and Python start from exactly the same samples. 1 s covers four of the
# undo blocks, so the joins between blocks get checked too.
def noise():
    rng = np.random.default_rng(4240)
    audio = 0.1 * rng.standard_normal(FS)
    return audio.astype(np.float32).astype(np.float64)


# Returns SciPy's Butterworth filter as second-order sections. A band-pass is a high-pass at the low edge followed
# by a low-pass at the high edge, each with the given order, the same way the C program builds it.
def butterworth(kind, order):
    highpass = signal.butter(order, HIGHPASS_HZ, "highpass", fs=FS, output="sos")
    lowpass = signal.butter(order, LOWPASS_HZ, "lowpass", fs=FS, output="sos")
    if kind == "HP":
        return highpass
    if kind == "LP":
        return lowpass
    return np.vstack([highpass, lowpass])


# Returns the audio with the filter undone, the way the C program does it. The audio goes through in blocks of
# N_FFT samples that overlap by GUARD samples on each side, with zeros past either end of the audio. Each block's
# spectrum gets multiplied by conj(H) / (|H|^2 + LAMBDA), and only the middle of each block is kept, so every
# sample comes from a block where it wasn't near the edge.
def undo_in_python(audio, sos):
    angles = 2 * np.pi * np.arange(N_FFT // 2 + 1) / N_FFT
    _, response = signal.sosfreqz(sos, worN=angles)
    inverse = np.conj(response) / (np.abs(response) ** 2 + LAMBDA)

    hop = N_FFT - 2 * GUARD
    output = np.zeros(len(audio))
    for offset in range(0, len(audio), hop):
        # Copy this block's samples, which start GUARD samples before the offset.
        block = np.zeros(N_FFT)
        first = max(0, offset - GUARD)
        end = min(len(audio), offset - GUARD + N_FFT)
        start_in_block = first - (offset - GUARD)
        block[start_in_block:start_in_block + end - first] = audio[first:end]

        # Undo the filter on this block, then keep its middle.
        recovered = np.fft.irfft(np.fft.rfft(block) * inverse, N_FFT)
        keep = min(hop, len(audio) - offset)
        output[offset:offset + keep] = recovered[GUARD:GUARD + keep]

    return output


# Runs the C program on audio in a new folder, and returns its output audio. The arguments follow its command
# line: type, family, order, high-pass cutoff, low-pass cutoff, input file, output file, and menu option. Each run
# gets its own folder, since a band-pass (option 3) writes an int.wav next to its files. It fails the check if the
# program didn't finish, or its output isn't a 32-bit float WAV with as many samples as the input.
def run_program(program, folder, audio, kind, order, option):
    folder.mkdir()
    sf.write(folder / "in.wav", audio, FS, subtype="FLOAT")

    arguments = [str(program), kind, "butter", str(order), str(HIGHPASS_HZ), str(LOWPASS_HZ),
                 "in.wav", "out.wav", str(option)]
    result = subprocess.run(arguments, cwd=folder, capture_output=True, text=True, check=False)
    output_text = result.stdout + result.stderr

    # The program returns 1 when it works, so 0 and 1 both count as finished.
    assert result.returncode in (0, 1), f"exit code {result.returncode}:\n{output_text}"
    assert (folder / "out.wav").exists(), f"no out.wav:\n{output_text}"

    info = sf.info(folder / "out.wav")
    assert info.subtype == "FLOAT"
    assert info.samplerate == FS
    assert info.frames == len(audio)

    output, _ = sf.read(folder / "out.wav", dtype="float64")
    return output


# Fails the check if the C output is further from the Python output than TOLERANCE times the Python output's peak.
# It prints the real difference, so a passing run still shows how close they were.
def assert_close(c_output, python_output, label):
    peak = np.max(np.abs(python_output))
    difference = np.max(np.abs(c_output - python_output)) / peak
    print(f"{label}: largest difference {difference:.2e} of the peak")
    assert difference < TOLERANCE, f"{label}: C and Python differ by {difference:.2e} of the peak"


# Tests: option 1 (low-pass) gives the same output as SciPy's Butterworth low-pass, at orders 1, 2, and 4.
# Why: the compensation filter undoes the filter our labels describe, and our Python filters are built to match
#   SciPy. If the C low-pass were even slightly different, the undoing would be off too.
def test_low_pass_matches_scipy(program, tmp_path):
    audio = noise()
    for order in ORDERS:
        c_output = run_program(program, tmp_path / f"order{order}", audio, "LP", order, 1)
        python_output = signal.sosfilt(butterworth("LP", order), audio)
        assert_close(c_output, python_output, f"low-pass order {order}")


# Tests: option 2 (high-pass) gives the same output as SciPy's Butterworth high-pass at the high-pass cutoff, at
#   orders 1, 2, and 4.
# Why: same as the low-pass. Right now this fails: main.c passes the low-pass cutoff (AI_fc_lp) to the high-pass
#   for option 2, so it filters at 5 kHz instead of 300 Hz. It passes once that's fixed. We only call options 4-6,
#   so it doesn't hold us up.
def test_high_pass_matches_scipy(program, tmp_path):
    audio = noise()
    for order in ORDERS:
        c_output = run_program(program, tmp_path / f"order{order}", audio, "HP", order, 2)
        python_output = signal.sosfilt(butterworth("HP", order), audio)
        assert_close(c_output, python_output, f"high-pass order {order}")


# Tests: option 3 (band-pass) gives the same output as SciPy's high-pass followed by its low-pass, each at the
#   given order, at orders 1, 2, and 4 per side.
# Why: our band-pass labels count the poles on both sides, and the C program counts each side, so this checks the
#   two really are the same filter when the order gets halved.
def test_band_pass_matches_scipy(program, tmp_path):
    audio = noise()
    for order in ORDERS:
        c_output = run_program(program, tmp_path / f"order{order}", audio, "BP", order, 3)
        python_output = signal.sosfilt(butterworth("BP", order), audio)
        assert_close(c_output, python_output, f"band-pass order {order}")


# Tests: options 4, 5, and 6 (undo a low-pass, high-pass, or band-pass) give the same output as the regularized
#   inverse written in Python, at orders 1, 2, and 4, on noise that went through that filter first.
# Why: this is the C vs Python check in our decisions. The evaluation undoes filters with the Python copy whenever
#   the C program isn't built, so the two have to agree for those results to stand for the C program.
def test_undo_matches_python(program, tmp_path):
    for kind, option in [("LP", 4), ("HP", 5), ("BP", 6)]:
        for order in ORDERS:
            sos = butterworth(kind, order)
            filtered = signal.sosfilt(sos, noise()).astype(np.float32).astype(np.float64)
            c_output = run_program(program, tmp_path / f"{kind}{order}", filtered, kind, order, option)
            python_output = undo_in_python(filtered, sos)
            assert_close(c_output, python_output, f"undo {kind} order {order}")


# Tests: a filter family other than butter stops the program with an error code, without writing out.wav.
# Why: the program only knows Butterworth filters. Anything else has to fail loudly, not leave an output that looks
#   like it worked.
def test_unknown_family_stops(program, tmp_path):
    sf.write(tmp_path / "in.wav", noise(), FS, subtype="FLOAT")
    arguments = [str(program), "LP", "cheby", "2", str(HIGHPASS_HZ), str(LOWPASS_HZ), "in.wav", "out.wav", "1"]
    result = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode not in (0, 1)
    assert not (tmp_path / "out.wav").exists()
