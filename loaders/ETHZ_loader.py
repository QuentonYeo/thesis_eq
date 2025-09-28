import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from seisbench.models.base import WaveformModel
from seisbench.data.base import BenchmarkDataset
from seisbench.generate.labeling import SupervisedLabeller

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding


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


def dump_metadata_to_csv(sbd: BenchmarkDataset, filename="metadata.csv"):
    """
    Dumps metadata values from the sbd object to a CSV file in /metadata/.
    Args:
        sbd: The seisbench dataset object with metadata attribute (assumed to be a DataFrame or dict-like).
        filename: Name of the CSV file to write (default: metadata.csv).
    """
    metadata_dir = os.path.join(os.path.dirname(__file__), "../metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    csv_path = os.path.join(metadata_dir, filename)

    metadata = getattr(sbd, "metadata", None)
    if metadata is None:
        raise ValueError("sbd object does not have a 'metadata' attribute.")

    # If pandas DataFrame
    try:
        import pandas as pd

        if isinstance(metadata, pd.DataFrame):
            metadata.to_csv(csv_path, index=False)
            return csv_path
    except ImportError:
        pass

    # If dict-like
    if isinstance(metadata, dict):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["key", "value"])
            for k, v in metadata.items():
                writer.writerow([k, v])
        return csv_path

    # If list of dicts
    if isinstance(metadata, list) and all(isinstance(item, dict) for item in metadata):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
        return csv_path

    raise TypeError(
        "Unsupported metadata format. Must be pandas DataFrame, dict, or list of dicts."
    )


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
        # print(
        #     f"[MagnitudeLabeller] mag: {mag}, onset: {onset}, label (nonzero count): {np.count_nonzero(label)}, label (unique): {np.unique(label)}"
        # )
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


class EQTransformerLabeller(SupervisedLabeller):
    """
    Custom labeller for EQTransformer that creates Detection, P, and S labels
    without the automatic Noise channel that ProbabilisticLabeller adds.
    """

    def __init__(self, phase_dict, sigma=30, key=("X", "y")):
        super().__init__(label_type="multi_label", dim=0, key=key)
        self.phase_dict = phase_dict
        self.sigma = sigma
        self.label_columns = list(phase_dict.keys())

    def label(self, X, metadata):
        length = X.shape[-1]

        # Initialize arrays for Detection, P, S
        detection = np.zeros(length, dtype=np.float32)
        p_phase = np.zeros(length, dtype=np.float32)
        s_phase = np.zeros(length, dtype=np.float32)

        # Process each phase pick
        for pick_key, phase_type in self.phase_dict.items():
            pick_sample = metadata.get(pick_key, np.nan)
            if not np.isnan(pick_sample) and 0 <= pick_sample < length:
                pick_sample = int(pick_sample)

                # Create gaussian around the pick
                samples = np.arange(length, dtype=np.float32)
                gaussian = np.exp(-0.5 * ((samples - pick_sample) / self.sigma) ** 2)

                # Add to appropriate channel
                if phase_type == "P":
                    p_phase += gaussian
                    detection += gaussian  # P picks contribute to detection
                elif phase_type == "S":
                    s_phase += gaussian
                    detection += gaussian  # S picks contribute to detection

        # Clip values to [0, 1]
        detection = np.clip(detection, 0, 1)
        p_phase = np.clip(p_phase, 0, 1)
        s_phase = np.clip(s_phase, 0, 1)

        # Stack as [Detection, P, S]
        y = np.stack([detection, p_phase, s_phase], axis=0)
        return y


def get_augmentation_eqt_custom(model):
    """Custom augmentation for EQTransformer using our custom labeller"""

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

    augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=3000,
            windowlen=9000,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(windowlen=6000, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        EQTransformerLabeller(phase_dict, sigma=30),
    ]
    return augmentations


def load_dataset(
    model: WaveformModel, type: str
) -> tuple[sbg.GenericGenerator, DataLoader, BenchmarkDataset]:

    # Load ETHZ @100Hz for sampling rate and use defined training splits
    # NOTE: These dataset are located in ~/.seisbench for transfer or removal

    data = sbd.ETHZ(
        sampling_rate=100
    )  # if gots memory for it use cache="trace" if need to redownload use force=True
    # data = sbd.MLAAPDE(sampling_rate=100, force=True)
    # data = sbd.GEOFON(sampling_rate=100)

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


def load_dataset_eqt(model: WaveformModel, type: str):
    """Load dataset for EQTransformer using custom labeller"""

    data = sbd.ETHZ(sampling_rate=100)
    train, dev, test = data.train_dev_test()

    batch_size = 256
    num_workers = 4

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
    ds_generator.add_augmentations(get_augmentation_eqt_custom(model))

    # Add magnitude labeller only if model has magnitude prediction
    if hasattr(model, "predict_magnitude") and model.predict_magnitude:
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
    # Plot Z, N, E waveforms
    channel_names = ["Z", "N", "E"]
    for i in range(sample["X"].shape[0]):
        axs[0].plot(sample["X"][i], label=channel_names[i])
    axs[0].set_ylabel("Waveform")
    axs[0].legend()

    # Plot P, S, Noise phase labels
    phase_names = ["P", "S", "Noise"]
    colors = ["tab:blue", "tab:green", "tab:orange"]
    for i in range(sample["y"].shape[0]):
        axs[1].plot(sample["y"][i], label=phase_names[i], color=colors[i])
    axs[1].set_ylabel("Phase Label")
    axs[1].legend()

    # Plot magnitude label
    if "magnitude" in sample:
        mag_data = sample["magnitude"]
        axs[2].plot(mag_data, color="tab:orange")
        axs[2].set_ylabel("Magnitude Label")
    else:
        axs[2].text(0.5, 0.5, "No magnitude label", ha="center", va="center")
    axs[2].set_xlabel("Sample Index")

    plot_magnitude_distribution(data)
    # dump_metadata_to_csv(data, "GEOFON_metadata.csv")

    plt.show()
