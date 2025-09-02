import numpy as np


class MagnitudeAfterPWave:
    """
    Adds a 'magnitude' label to each sample, set to the event's magnitude after the first P pick.
    """

    def __init__(self, metadata):
        self.magnitudes = metadata["source_magnitude"].values

    def __call__(self, sample):
        # Try to get event index from sample
        event_idx = sample.get("event_index", None)
        if event_idx is None:
            # If not available, skip augmentation
            sample["magnitude"] = np.zeros_like(sample["y"][0])
            return sample

        mag = self.magnitudes[event_idx]
        # Find first P pick (assuming P is index 0 in y)
        p_pick_idx = np.argmax(sample["y"][0] > 0)
        mag_label = np.zeros_like(sample["y"][0])
        if sample["y"][0, p_pick_idx] > 0:
            mag_label[p_pick_idx:] = mag
        sample["magnitude"] = mag_label
        return sample
