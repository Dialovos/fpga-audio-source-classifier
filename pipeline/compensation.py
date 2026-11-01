"""
Undo known Butterworth filters without clipping the recovered audio.

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


# Returns finite mono audio. It stops with a ValueError if the audio is empty,
# non-finite, or has more than one channel.
def _samples(audio) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float64)

    if samples.ndim != 1 or len(samples) == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("audio must be non-empty finite mono audio")

    return samples


# Returns the filter for a plain supported spec. It stops with a ValueError if
# the spec has extra keys, a different family, or an unsupported order.
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


# Returns the recovered audio as float64 without clipping. It stops with a
# ValueError if the audio, filter spec, or FFT settings are invalid.
def python_inverse(audio, sample_rate, spec, lam=0.001, n_fft=65536, guard=2048) -> np.ndarray:
    samples = _samples(audio)
    sos = _filter(spec, sample_rate)

    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be positive and finite")

    if not isinstance(n_fft, int) or n_fft < 2 or n_fft % 2:
        raise ValueError("n_fft must be a positive even integer")

    if not isinstance(guard, int) or not 0 <= 2 * guard < n_fft:
        raise ValueError("guard must leave a positive hop inside n_fft")

    angles = 2 * np.pi * np.arange(n_fft // 2 + 1) / n_fft
    _, response = signal.sosfreqz(sos, worN=angles)
    inverse = np.conj(response) / (np.abs(response) ** 2 + lam)

    hop = n_fft - 2 * guard
    output = np.empty(len(samples), dtype=np.float64)

    for offset in range(0, len(samples), hop):
        block = np.zeros(n_fft)
        first = max(0, offset - guard)
        end = min(len(samples), offset - guard + n_fft)
        block_start = first - (offset - guard)
        block[block_start:block_start + end - first] = samples[first:end]

        spectrum = np.fft.rfft(block)
        spectrum *= inverse
        recovered = np.fft.irfft(spectrum, n_fft)

        keep = min(hop, len(samples) - offset)
        output[offset:offset + keep] = recovered[guard:guard + keep]

    return output


# ********************************
# C inverse and choosing a method
# ********************************


# Returns the C arguments for a plain spec. A band-pass uses half the total
# order because the C program counts the poles on each side separately.
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


# Returns the C program's recovered audio. It stops with a ValueError, including
# the program's output, if the command fails or its WAV is invalid. Each call
# gets its own system temporary folder so the program's int.wav is not shared.
def c_inverse(audio, sample_rate, spec, command) -> np.ndarray:
    samples = _samples(audio)
    _filter(spec, sample_rate)

    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty list")

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
        sf.write(path / "in.wav", samples, sample_rate, subtype="FLOAT")

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

        if result.returncode not in (0, 1) or not (path / "out.wav").is_file():
            raise ValueError(f"command failed or out.wav is missing (exit {result.returncode}): {output}")

        try:
            info = sf.info(path / "out.wav")
            recovered, rate = sf.read(path / "out.wav", dtype="float64")
        except (sf.LibsndfileError, OSError) as error:
            raise ValueError(f"out.wav could not be read: {output}\n{error}") from error

        if info.subtype != "FLOAT" or info.channels != 1 or rate != sample_rate or len(recovered) != len(samples):
            raise ValueError(f"out.wav must be FLOAT mono with the input rate and length: {output}")

        if not np.all(np.isfinite(recovered)):
            raise ValueError(f"out.wav must be finite: {output}")

        return recovered


# Returns recovered audio and the inverse settings. It stops with a ValueError
# if the method, spec, audio, or C output is invalid.
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
