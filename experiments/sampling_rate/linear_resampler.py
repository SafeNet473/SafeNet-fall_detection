"""Isolated causal interpolation experiment, never installed in the detector.

Uniform 208 -> 200 Hz uses exact integer phase: output k corresponds to input
position k*26/25. Emit only when the bracketing input sample has arrived.
Timestamps are acquisition timestamps, not host arrival/interrupt timestamps.
No extrapolation, no dropping every 26th sample, no antialiasing guarantee.
"""
from __future__ import annotations
import numpy as np


class Linear208To200:
    def __init__(self):
        self.previous = None
        self.input_index = -1
        self.output_index = 0

    def push(self, sample):
        current = np.asarray(sample, dtype=float)
        self.input_index += 1
        emitted = []
        if self.previous is None:
            self.previous = current.copy()
            self.output_index = 1
            return [(0., current.copy(), 0.)]
        while self.output_index * 26 <= self.input_index * 25:
            phase_numerator = self.output_index * 26 - (self.input_index - 1) * 25
            assert 0 <= phase_numerator <= 25
            fraction = phase_numerator / 25.
            value = self.previous + fraction * (current - self.previous)
            output_time = self.output_index / 200.
            arrival_time = self.input_index / 208.
            emitted.append((output_time, value.copy(), arrival_time))
            self.output_index += 1
        self.previous = current.copy()
        return emitted


def run(raw208):
    state = Linear208To200()
    output = []
    arrivals = []
    for sample in raw208:
        for time, value, arrival in state.push(sample):
            output.append(value)
            arrivals.append(arrival - time)
    return np.asarray(output), np.asarray(arrivals)
