#!/usr/bin/env python3
import time
import numpy as np

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
