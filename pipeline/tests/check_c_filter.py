"""
Checks for the C filter program in pipeline/reversable_filters/.

These build the C program and run it the way the rest of the project will, as a program on 32-bit float WAV
files. Options 1-3 (applying a filter) get compared with SciPy's Butterworth filters, and options 4-6 (undoing a
filter) with the same regularized inverse written in Python. The undo also gets checked against the original signal
and against an undo done in one piece, and the program gets bad calls to see what it does. The C code itself isn't
changed.

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

# Lower cutoffs, for the check on how far the undo reaches. A lower cutoff makes the inverse filter last longer.
LOW_HIGHPASS_HZ = 50.0
LOW_LOWPASS_HZ = 500.0

# The C program counts the order on each side of a band-pass, and only takes these.
ORDERS = [1, 2, 4]

# How close the C output has to be to Python, as a share of the signal's peak. The C program saves 32-bit floats,
# which round at about 6e-8 of the peak, so this leaves plenty of room for that and nothing for a wrong filter.
TOLERANCE = 1e-5

# For the round trip (filter, then undo, then compare with the original). The signal is 0.25 s, so it fits in one
# block and has no joins. The first and last ROUND_TRIP_EDGE samples don't count, because the inverse filter needs
# more samples around each one than the file has at its ends. The signal only has frequencies where the filter's
# gain squared is at least PASSBAND_GAIN. There the regularization costs under 0.2% of the signal, so
# ROUND_TRIP_TOLERANCE (1% of the peak) leaves room for that and nothing for a wrong filter.
ROUND_TRIP_SAMPLES = FS // 4
ROUND_TRIP_EDGE = 8192
PASSBAND_GAIN = 0.5
ROUND_TRIP_TOLERANCE = 1e-2


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
def butterworth(kind, order, highpass_hz=HIGHPASS_HZ, lowpass_hz=LOWPASS_HZ):
    highpass = signal.butter(order, highpass_hz, "highpass", fs=FS, output="sos")
    lowpass = signal.butter(order, lowpass_hz, "lowpass", fs=FS, output="sos")
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


# Returns the audio with the filter undone in one piece. It's the same regularized inverse as undo_in_python, but
# the whole file goes through a single FFT (padded with zeros to at least twice its length), so there are no block
# joins. The C program's blocks should give the same answer, as long as the inverse filter dies out within GUARD
# samples.
def undo_in_one_piece(audio, sos):
    n_fft = 1 << (2 * len(audio) - 1).bit_length()
    angles = 2 * np.pi * np.arange(n_fft // 2 + 1) / n_fft
    _, response = signal.sosfreqz(sos, worN=angles)
    inverse = np.conj(response) / (np.abs(response) ** 2 + LAMBDA)
    return np.fft.irfft(np.fft.rfft(audio, n_fft) * inverse, n_fft)[:len(audio)]


# Returns noise at an RMS of 0.1 that only has the frequencies the filter lets through (where its gain squared is at
# least PASSBAND_GAIN), rounded to 32-bit floats like noise(). The undo can only bring back what the filter let
# through, so this is the signal to ask it to bring back.
def passband_noise(sos, samples):
    angles = 2 * np.pi * np.arange(samples // 2 + 1) / samples
    _, response = signal.sosfreqz(sos, worN=angles)
    passed = np.abs(response) ** 2 >= PASSBAND_GAIN
    rng = np.random.default_rng(4240)
    audio = np.fft.irfft(np.fft.rfft(rng.standard_normal(samples)) * passed, samples)
    audio *= 0.1 / audio.std()
    return audio.astype(np.float32).astype(np.float64)


# Runs the C program on audio in a new folder, and returns its output audio. The arguments follow its command
# line: type, family, order, high-pass cutoff, low-pass cutoff, input file, output file, and menu option. The
# cutoffs are the readme's unless the check passes others. Each run gets its own folder, since a band-pass
# (option 3) writes an int.wav next to its files. It fails the check if the program didn't finish, or its output
# isn't a 32-bit float WAV with as many samples as the input.
def run_program(program, folder, audio, kind, order, option, highpass_hz=HIGHPASS_HZ, lowpass_hz=LOWPASS_HZ):
    folder.mkdir()
    sf.write(folder / "in.wav", audio, FS, subtype="FLOAT")

    arguments = [str(program), kind, "butter", str(order), str(highpass_hz), str(lowpass_hz),
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


# Returns how far the C output is from the reference, as a share of the reference's peak, and prints it so a
# passing run still shows how close they were.
def difference_of(c_output, reference, label):
    difference = np.max(np.abs(c_output - reference)) / np.max(np.abs(reference))
    print(f"{label}: largest difference {difference:.2e} of the peak")
    return difference


# Fails the check if the C output is further from the Python output than TOLERANCE times the Python output's peak.
def assert_close(c_output, python_output, label):
    difference = difference_of(c_output, python_output, label)
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


# Tests: options 4, 5, and 6 (undo a low-pass, high-pass, or band-pass) bring back the original signal, at orders 1,
#   2, and 4. The signal is noise inside the filter's passband, run through SciPy's filter first.
# Why: test_undo_matches_python only shows the C program agrees with a Python copy of the same method. This checks
#   the method itself: the filter goes on in SciPy, comes off in C, and the result gets compared with what went in.
def test_undo_brings_back_the_signal(program, tmp_path):
    problems = []
    for kind, option in [("LP", 4), ("HP", 5), ("BP", 6)]:
        for order in ORDERS:
            sos = butterworth(kind, order)
            original = passband_noise(sos, ROUND_TRIP_SAMPLES)
            filtered = signal.sosfilt(sos, original).astype(np.float32).astype(np.float64)
            c_output = run_program(program, tmp_path / f"{kind}{order}", filtered, kind, order, option)

            middle = slice(ROUND_TRIP_EDGE, -ROUND_TRIP_EDGE)
            label = f"round trip {kind} order {order}"
            difference = difference_of(c_output[middle], original[middle], label)
            if difference >= ROUND_TRIP_TOLERANCE:
                problems.append(f"{label}: off by {difference:.2e} of the peak")
    assert not problems, "the undo doesn't bring the signal back:\n" + "\n".join(problems)


# Tests: options 4, 5, and 6 give the same output as undoing the filter in one piece, at orders 1, 2, and 4, at the
#   readme's cutoffs and at lower ones.
# Why: the C program works in blocks and throws away GUARD samples at each end of every block. That only works if
#   the inverse filter's effect dies out within GUARD samples, and a lower cutoff makes it last longer.
#   test_undo_matches_python can't show a problem here, since the Python copy uses the same blocks. Undoing in one
#   piece has no joins to get wrong.
#   Right now this fails for the high-pass and band-pass undo (the low-pass undo is fine). Their inverse filter lasts
#   longer than the guard, so each block comes out wrong near its ends, worst at the joins: 0.6% to 1.4% of the peak
#   at the readme's cutoffs, and up to 24% at the lower ones. The Python copy has the same errors with the same
#   guard, and they go away in it when the guard is much larger (32768, with blocks of 262144), so the guard is the
#   cause. This should pass once the C program's guard is large enough.
def test_undo_matches_one_piece(program, tmp_path):
    audio = noise()
    problems = []
    for highpass_hz, lowpass_hz in [(HIGHPASS_HZ, LOWPASS_HZ), (LOW_HIGHPASS_HZ, LOW_LOWPASS_HZ)]:
        for kind, option in [("LP", 4), ("HP", 5), ("BP", 6)]:
            for order in ORDERS:
                sos = butterworth(kind, order, highpass_hz, lowpass_hz)
                filtered = signal.sosfilt(sos, audio).astype(np.float32).astype(np.float64)
                folder = tmp_path / f"{kind}{order}-{highpass_hz:g}-{lowpass_hz:g}"
                c_output = run_program(program, folder, filtered, kind, order, option, highpass_hz, lowpass_hz)

                label = f"undo {kind} order {order} at {highpass_hz:g} and {lowpass_hz:g} Hz"
                difference = difference_of(c_output, undo_in_one_piece(filtered, sos), label)
                if difference >= TOLERANCE:
                    problems.append(f"{label}: off by {difference:.2e} of the peak")
    assert not problems, "the undo has errors at the block joins:\n" + "\n".join(problems)


# Tests: a call the program can't carry out (the input file is missing, empty, or not a WAV, or the menu option isn't
#   one it has) leaves no out.wav behind.
# Why: the program returns 1 whether it did the job or not, so out.wav not being there is the only way a script can
#   tell the call failed.
def test_bad_calls_leave_no_output(program, tmp_path):
    calls = [
        ("input missing", None, 1),
        ("input empty", b"", 1),
        ("input not a WAV", b"this is text, not a WAV file. " * 10, 1),
        ("option 0", "audio", 0),
        ("option 7", "audio", 7),
        ("option 9", "audio", 9),
    ]
    problems = []
    for label, content, option in calls:
        folder = tmp_path / label.replace(" ", "-")
        folder.mkdir()
        if content == "audio":
            sf.write(folder / "in.wav", noise(), FS, subtype="FLOAT")
        elif content is not None:
            (folder / "in.wav").write_bytes(content)

        arguments = [str(program), "LP", "butter", "2", str(HIGHPASS_HZ), str(LOWPASS_HZ), "in.wav", "out.wav",
                     str(option)]
        subprocess.run(arguments, cwd=folder, capture_output=True, text=True, check=False, timeout=10)
        if (folder / "out.wav").exists():
            problems.append(label)
    assert not problems, "these calls still wrote an out.wav: " + ", ".join(problems)


# Tests: an order the program doesn't have (0, 3, or 8) stops with an error code, without writing out.wav.
# Why: the readme says 1, 2, and 4 are the only orders it supports. Anything else has to fail loudly, not leave an
#   output that looks like it worked.
#   Right now this fails: main.c doesn't check the order, and the filter code only sets up orders 1, 2, and 4, so the
#   program exits with 1 and writes an out.wav anyway. It passes once the order gets checked.
def test_unsupported_order_stops(program, tmp_path):
    problems = []
    for order in [0, 3, 8]:
        folder = tmp_path / f"order{order}"
        folder.mkdir()
        sf.write(folder / "in.wav", noise(), FS, subtype="FLOAT")

        arguments = [str(program), "LP", "butter", str(order), str(HIGHPASS_HZ), str(LOWPASS_HZ), "in.wav",
                     "out.wav", "1"]
        result = subprocess.run(arguments, cwd=folder, capture_output=True, text=True, check=False, timeout=10)
        wrote = (folder / "out.wav").exists()
        if result.returncode in (0, 1) or wrote:
            state = "written" if wrote else "not written"
            problems.append(f"order {order}: exit code {result.returncode}, out.wav {state}")
    assert not problems, "an unsupported order didn't stop the program:\n" + "\n".join(problems)


# Tests: calling the program with the wrong number of arguments stops with an error code, not a wait for input.
# Why: with anything but 9 arguments, main.c starts its menu for a person at a keyboard. A script that leaves one
#   out, or adds one, would hang.
#   Right now this fails: the program prints the menu again and again until it's killed. Its output goes to
#   DEVNULL here, since it would fill the memory of a captured pipe in seconds.
def test_wrong_argument_count_stops(program, tmp_path):
    arguments = [str(program), "LP", "butter", "2", str(HIGHPASS_HZ), str(LOWPASS_HZ), "in.wav", "out.wav"]
    try:
        result = subprocess.run(arguments, cwd=tmp_path, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, check=False, timeout=5)
    except subprocess.TimeoutExpired:
        pytest.fail("the program was still running 5 s after a call with 8 arguments, so it's waiting at its menu")
    assert result.returncode not in (0, 1)
