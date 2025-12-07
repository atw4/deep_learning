import numpy as np
from Utility.Timer import Timer

class Benchmark:
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
