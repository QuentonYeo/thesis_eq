import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from seisbench.models.base import WaveformModel
from seisbench.data.base import BenchmarkDataset
from seisbench.generate.labeling import SupervisedLabeller


# Only training for S and P picks, map the labels
phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}


class MagnitudeLabeller(SupervisedLabeller):
    """
    Labeller for magnitude regression: sets all values to zero before first P pick,
    and to the event's source magnitude after the first P pick.
    """

    def __init__(
        self,
        phase_dict=phase_dict,
        magnitude_column="source_magnitude",
        key=("X", "magnitude"),
    ):
        super().__init__(label_type="multi_label", dim=1, key=key)
        self.phase_dict = phase_dict
        self.magnitude_column = magnitude_column
        self.label_columns = list(phase_dict.keys()) + [magnitude_column]

    def label(self, X, metadata):
        length = X.shape[-1]
        mag = metadata.get(self.magnitude_column, 0.0)
        # Find the earliest pick time from phase_dict keys
        pick_times = []
        for pick_key in self.phase_dict.keys():
            pick = metadata.get(pick_key, np.nan)
            if not np.isnan(pick):
                pick_times.append(pick)
        if pick_times:
            onset = int(min(pick_times))
        else:
            onset = None
        label = np.zeros(length, dtype=np.float32)
        if onset is not None and onset < length:
            label[onset:] = mag
        # Debug print
        print(
            f"[MagnitudeLabeller] mag: {mag}, onset: {onset}, label (nonzero count): {np.count_nonzero(label)}, label (unique): {np.unique(label)}"
        )
        return label


def get_augmentation(model: WaveformModel):
    """Define training and validation generator with the following augmentations:

    - Long window around pick
    - Random window of 3001 samples (Phasenet input length)
    - Change datatype to float32 for pytorch
    - Probablistic label
    """
    augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=3000,
            windowlen=6000,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(windowlen=3001, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict, model_labels=model.labels, sigma=30, dim=0
        ),
    ]
    return augmentations


def load_dataset(
    model: WaveformModel, type: str
) -> tuple[sbg.GenericGenerator, DataLoader, BenchmarkDataset]:

    # Load ETHZ @100Hz for sampling rate and use defined training splits
    data = sbd.ETHZ(sampling_rate=100)  # if gots memory for it use cache="trace"
    train, dev, test = data.train_dev_test()

    # Dataloader params
    batch_size = 512
    num_workers = 8

    if type == "train":
        train.preload_waveforms()
        dataset = train
    elif type == "dev":
        dev.preload_waveforms()
        dataset = dev
    else:
        test.preload_waveforms()
        dataset = test

    ds_generator = sbg.GenericGenerator(dataset)
    ds_generator.add_augmentations(get_augmentation(model))
    ds_generator.add_augmentations([MagnitudeLabeller()])

    loader = DataLoader(
        ds_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )

    return ds_generator, loader, data


def plot_magnitude_distribution(data: BenchmarkDataset) -> None:
    magnitudes = data.metadata["source_magnitude"]

    # Plot histogram: frequency vs magnitude bins (0-9, step 0.5)
    bins = [x * 0.5 for x in range(19)]  # 0, 0.5, ..., 10
    plt.figure()
    plt.hist(magnitudes, bins=bins, edgecolor="black")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.title("Magnitude Distribution in DummyDataset")
    plt.xticks(bins)
    plt.show()


if __name__ == "__main__":
    # Load standard Phasenet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    model.to_preferred_device(verbose=True)

    train_generator, _, data = load_dataset(model, "train")

    # Example training input
    sample = train_generator[np.random.randint(len(train_generator))]

    print(sample)

    fig = plt.figure(figsize=(15, 12))
    axs = fig.subplots(
        3, 1, sharex=True, gridspec_kw={"hspace": 0.2, "height_ratios": [3, 1, 1]}
    )
    axs[0].plot(sample["X"].T)
    axs[0].set_ylabel("Waveform")
    axs[1].plot(sample["y"].T)
    axs[1].set_ylabel("Phase Label")
    if "magnitude" in sample:
        mag_data = sample["magnitude"]
        axs[2].plot(mag_data, color="tab:orange")
        axs[2].set_ylabel("Magnitude Label")
    else:
        axs[2].text(0.5, 0.5, "No magnitude label", ha="center", va="center")
    axs[2].set_xlabel("Sample Index")

    # plot_magnitude_distribution(data)

    plt.show()
