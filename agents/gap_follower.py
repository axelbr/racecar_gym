import math
from typing import Dict, Tuple
import numpy as np
from scipy.signal import medfilt


class GapFollower:

    def __init__(self):
        self.threshold = 0.1

    def preprocess_lidar(self, ranges, kernel_size=5):
        # Step 1: interpolate nan values
        proc_ranges = np.array(ranges)
        nans = np.isnan(proc_ranges)
        nan_idx = np.where(nans)
        x = np.where(~nans)[0]
        y = proc_ranges[~nans]
        proc_ranges[nan_idx] = np.interp(nan_idx, x, y)

        # Step 2: apply a median filter to the interpolated values
        proc_ranges = medfilt(proc_ranges, kernel_size)
        return proc_ranges

    def find_max_gap(self, free_space_ranges, min_distance):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        gaps = np.hstack(([False], free_space_ranges >= min_distance + 0.1, [False]))
        gap_indices = np.where(np.diff(gaps))[0].reshape(-1, 2)
        if gap_indices.size != 0:
            largest_gap = gap_indices[np.argmax(np.diff(gap_indices))]
            return largest_gap
        else:
            return np.ndarray([0, free_space_ranges.size - 1])

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        center_index = int((start_i + end_i) / 2)
        return center_index

    def action(self, observation: Dict[str, np.ndarray]) -> Tuple[float, float]:
        scan = observation['lidar']
        if len(scan.shape) > 1:
            scan = scan.reshape(scan.shape[1])
        proc_ranges = self.preprocess_lidar(scan, kernel_size=3)
        min_index, max_index = 0, 1080
        proc_ranges = proc_ranges[min_index:max_index]

        # Find closest point to LiDAR
        min_distance = np.min(proc_ranges)

        # Find max length gap
        if len(proc_ranges) < 1:
            return (0, 0)
        gap = self.find_max_gap(free_space_ranges=proc_ranges, min_distance=min_distance)

        if len(gap) < 1:
            return (0, 0)
        # Find the best point in the gap
        best_point = min_index + self.find_best_point(start_i=gap[0], end_i=gap[1], ranges=proc_ranges)

        # Publish Drive message
        angle = (-math.pi/2 + best_point * math.pi / 1080)
        angle = math.copysign(min(1, abs(angle)), angle)

        command = angle / 0.42

        return np.random.normal(loc=0.1, scale=0.0), np.random.normal(loc=command, scale=0.0)

    def __call__(self, obs, *args):
        return np.expand_dims(np.array(self.action(obs)), 0), obs
