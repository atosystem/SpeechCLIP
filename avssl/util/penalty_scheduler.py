import numpy as np


class PenaltyScheduler:
    def __init__(self, weights, keypoints):
        assert len(weights) == len(keypoints)

        self.weights = weights
        self.keypoints = keypoints
        self.value = self.weights[0]
        self.update(0)

    def update(self, global_step):
        if global_step >= self.keypoints[-1]:
            self.value = self.weights[-1]

        elif global_step <= self.keypoints[0]:
            self.value = self.weights[0]
        else:
            idx = np.searchsorted(self.keypoints, global_step, side="right")
            # between idx-1, idx
            ratio = (global_step - self.keypoints[idx - 1]) / (
                self.keypoints[idx] - self.keypoints[idx - 1]
            )
            self.value = ratio * self.weights[idx] + (1 - ratio) * self.weights[idx - 1]

    def get_value(self):
        return self.value


if __name__ == "__main__":
    ps = PenaltyScheduler(weights=[0, 1, 1, 5], keypoints=[0, 100, 500, 1000])

    ps.update(0)
    print(ps.value)
    ps.update(999)
    print(ps.value)
