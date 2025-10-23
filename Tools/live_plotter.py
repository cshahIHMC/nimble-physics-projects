#!/usr/bin/env python3
"""
LivePlotter3D
=============
A simple reusable class for live plotting 3-axis data (e.g., accelerometer, gyroscope, magnetometer).

Usage:
------
from live_plotter import LivePlotter3D
import numpy as np, time

plotter = LivePlotter3D(window=5.0, title="Live Accelerometer Data")

while True:
    acc = np.random.randn(3)  # example 1x3 vector
    plotter.update(acc)
    time.sleep(0.01)  # simulate ~100 Hz stream
"""

import matplotlib.pyplot as plt
import numpy as np
import time


class LivePlotter3D:
    def __init__(self, window: float = 5.0, title: str = "Live 3-Axis Data"):
        """
        Initialize the live plotter.

        Args:
            window (float): time window in seconds to display
            title (str): figure title
        """
        self.window = window
        self.start_time = time.time()
        self.times = []
        self.data = np.zeros((0, 3))  # stores [x,y,z] rows

        # --- Matplotlib setup ---
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        self.fig.suptitle(title)

        self.lines = []
        colors = ["r", "g", "b"]
        labels = ["X", "Y", "Z"]
        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], color=colors[i], label=labels[i])
            self.lines.append(line)
            ax.set_xlim(0, window)
            ax.set_ylim(-2, 2)
            ax.set_ylabel(labels[i])
            ax.legend(loc="upper right")

        self.axes[-1].set_xlabel("Time (s)")
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])

    def update(self, vec: np.ndarray):
        """
        Update the plot with a new 1x3 data vector.

        Args:
            vec (np.ndarray): 1x3 array-like [x, y, z]
        """
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec, dtype=float)
        if vec.shape != (3,):
            raise ValueError("Input must be a 1x3 vector (np.ndarray of shape (3,))")

        now = time.time() - self.start_time
        self.times.append(now)
        self.data = np.vstack([self.data, vec])

        # Keep only last 'window' seconds of data
        t_arr = np.array(self.times)
        mask = t_arr > (now - self.window)
        self.times = list(t_arr[mask])
        self.data = self.data[mask, :]

        # Update lines
        for i in range(3):
            self.lines[i].set_data(self.times, self.data[:, i])
            self.axes[i].set_xlim(max(0, self.times[0] if self.times else 0),
                                  max(self.window, now))
            # Adjust y-limits dynamically
            y_min, y_max = np.min(self.data[:, i]), np.max(self.data[:, i])
            span = y_max - y_min
            if span < 1e-6: span = 1.0
            self.axes[i].set_ylim(y_min - 0.1 * span, y_max + 0.1 * span)

        plt.pause(0.001)
