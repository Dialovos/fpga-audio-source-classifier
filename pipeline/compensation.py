"""
Undoes a known Butterworth filter, to bring back what the filter turned down.

There are 2 ways to do it here. The Python inverse is our own copy of the math in Adam's C filter, and it keeps the
recovered audio's peaks even when they go past full scale. The C inverse runs Adam's real C program. That program reads
& writes 16-bit WAV files, so its recovered audio gets clipped at full scale.

This file has three groups:
    * Shared checks
    * Python inverse
    * C inverse and choosing a method
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from pipeline.filters import design_filter

# ********************************
# Shared checks
# ********************************


# Gives back the audio as a numpy array after checking it. It stops with a ValueError if the audio is empty, has more
# than 1 channel, or has values that aren't real numbers (like infinity).
def _samples(audio) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float64)

    if samples.ndim != 1 or len(samples) == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("audio must be non-empty finite mono audio")

    return samples


# Builds the filter for a spec, after checking the spec is one the inverse can undo: Butterworth only, with nothing
# extra in the spec, at order 1, 2 or 4 (2, 4 or 8 for a band-pass). It stops with a ValueError if not.
def _filter(spec, sample_rate) -> np.ndarray:
    if not isinstance(spec, dict) or set(spec) != {"type", "family", "order", "cutoff"}:
        raise ValueError("spec must contain only type, family, order, cutoff")

    if spec["family"] != "butter":
        raise ValueError("spec family must be butter")

    orders = (1, 2, 4)

    if spec["type"] == "bandpass":
        orders = (2, 4, 8)

    if spec["order"] not in orders:
        raise ValueError("spec order is not supported by the C inverse")

    return design_filter(spec, sample_rate)


# ********************************
# Python inverse
# ********************************


# Undoes the filter in Python & gives back the recovered audio, without clipping it. It works through the audio a block
# at a time: take the block's spectrum (FFT), turn each frequency back up by as much as the filter turned it down, and
# turn it back into audio. lam keeps it from boosting without limit where the filter cut the deepest, n_fft is the block
# size & guard is the extra audio on each side of a block that gets thrown away, so the blocks join up cleanly. It stops
# with a ValueError if the audio, the spec, or those settings are bad.
def python_inverse(audio, sample_rate, spec, lam=0.001, n_fft=65536, guard=2048) -> np.ndarray:
    samples = _samples(audio)
    sos = _filter(spec, sample_rate)

    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be positive and finite")

    if not isinstance(n_fft, int) or n_fft < 2 or n_fft % 2:
        raise ValueError("n_fft must be a positive even integer")

    if not isinstance(guard, int) or not 0 <= 2 * guard < n_fft:
        raise ValueError("guard must leave a positive hop inside n_fft")

    # What the filter does at each of the FFT's frequencies (response), and the undo for it (inverse). Dividing by
    # |response|^2 + lam and not just |response|^2 is what keeps the boost from blowing up.
    angles = 2 * np.pi * np.arange(n_fft // 2 + 1) / n_fft
    _, response = signal.sosfreqz(sos, worN=angles)
    inverse = np.conj(response) / (np.abs(response) ** 2 + lam)

    # Each block overlaps its neighbors by `guard` samples on both sides. We only keep the middle `hop` samples.
    hop = n_fft - 2 * guard
    output = np.empty(len(samples), dtype=np.float64)

    for offset in range(0, len(samples), hop):
        block = np.zeros(n_fft)
        first = max(0, offset - guard)
        end = min(len(samples), offset - guard + n_fft)
        block_start = first - (offset - guard)
        block[block_start:block_start + end - first] = samples[first:end]

        # Into frequencies, apply the undo, and back into audio.
        spectrum = np.fft.rfft(block)
        spectrum *= inverse
        recovered = np.fft.irfft(spectrum, n_fft)

        keep = min(hop, len(samples) - offset)
        output[offset:offset + keep] = recovered[guard:guard + keep]

    return output


# ********************************
# C inverse and choosing a method
# ********************************


# Turns a filter spec into the arguments Adam's C program expects. A band-pass passes half its order, since the C
# program counts the poles on each side by itself.
def _arguments(spec) -> list[str]:
    kind = spec["type"]
    order = spec["order"]
    hp = "0"
    lp = "0"

    if kind == "bandpass":
        name = "BP"
        option = "6"
        order //= 2
        hp = repr(float(spec["cutoff"][0]))
        lp = repr(float(spec["cutoff"][1]))
    elif kind == "highpass":
        name = "HP"
        option = "5"
        hp = repr(float(spec["cutoff"]))
    else:
        name = "LP"
        option = "4"
        lp = repr(float(spec["cutoff"]))

    return [
        name,
        "butter",
        str(order),
        hp,
        lp,
        "in.wav",
        "out.wav",
        option,
    ]


# Undoes the filter with Adam's C program & gives back the recovered audio. The program reads & writes 48 kHz 16-bit WAV
# files, so the audio goes in rounded to 16-bit steps, and anything the undo pushes past full scale comes back clipped.
# Each call gets its own temporary folder, so 2 runs never share the program's int.wav file. It stops with a ValueError
# (with the program's output in it) if the audio is past full scale, the command fails, or the WAV it wrote is bad.
def c_inverse(audio, sample_rate, spec, command) -> np.ndarray:
    samples = _samples(audio)
    _filter(spec, sample_rate)

    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty list")

    if np.max(np.abs(samples)) > 1:
        raise ValueError("audio must be within full scale for the C program's 16-bit WAV")

    # The C program reads each sample as its 16-bit code / 32768, so save the codes themselves. A sample at exactly full
    # scale rounds to 32768, one past the highest code, so it gets the highest code.
    codes = np.clip(np.round(samples * 32768), -32768, 32767).astype(np.int16)

    absolute_command = []

    for item in command:
        path = Path(item)
        argument = str(item)

        if path.is_file():
            argument = str(path.resolve())

        absolute_command.append(argument)

    arguments = absolute_command + _arguments(spec)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        sf.write(path / "in.wav", codes, sample_rate, subtype="PCM_16")

        try:
            result = subprocess.run(
                arguments,
                cwd=folder,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise ValueError(f"command could not run: {error}") from error

        output = result.stdout + result.stderr

        if result.returncode != 0 or not (path / "out.wav").is_file():
            raise ValueError(f"command failed or out.wav is missing (exit {result.returncode}): {output}")

        try:
            info = sf.info(path / "out.wav")
            recovered, rate = sf.read(path / "out.wav", dtype="int16")
        except (sf.LibsndfileError, OSError) as error:
            raise ValueError(f"out.wav could not be read: {output}\n{error}") from error

        if info.subtype != "PCM_16" or info.channels != 1 or rate != sample_rate or len(recovered) != len(samples):
            raise ValueError(f"out.wav must be 16-bit PCM mono with the input rate and length: {output}")

        return recovered / 32768


# The function other code calls to undo a filter. method picks the Python inverse ("python") or the C program ("c", with
# command set to how to run it). It gives back the recovered audio & a record of the settings it used. It stops with a
# ValueError if the method, the spec, the audio, or the C program's output is bad.
def compensate(audio, sample_rate, spec, method="python", command=None) -> tuple[np.ndarray, dict]:
    _filter(spec, sample_rate)

    if method == "python":
        output = python_inverse(audio, sample_rate, spec)
    elif method == "c":
        output = c_inverse(audio, sample_rate, spec, command)
    else:
        raise ValueError("method must be python or c")

    record = {
        "method": method,
        "spec": dict(spec),
        "lam": 0.001,
        "n_fft": 65536,
        "guard": 2048,
    }
    record["peak"] = float(np.max(np.abs(output)))

    return output, record
