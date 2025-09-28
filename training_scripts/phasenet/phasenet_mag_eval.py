from models import phasenet_mag as PhaseNet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from tqdm import tqdm
import numpy as np

from loaders import ETHZ_loader as ETHZ

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PhaseNet model with magnitude prediction"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint file"
    )
    args = parser.parse_args()

    # Load PhaseNet with magnitude prediction
    model = PhaseNet.VariableLengthPhaseNet(
        in_samples=3001,
        phases="PSN",
        norm="std",
        default_args={"blinding": (200, 200)},
        predict_magnitude=True,
        magnitude_label="M",
    )
    model.to_preferred_device(verbose=True)

    # Load weights from specified checkpoint
    state_dict = torch.load(args.model_path, map_location=model.device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {args.model_path}")

    test_generator, _, _ = ETHZ.load_dataset(model, "test")

    # Visual check: plot a random test sample and prediction
    sample = test_generator[np.random.randint(len(test_generator))]
    fig = plt.figure(figsize=(15, 12))
    axs = fig.subplots(
        4, 1, sharex=True, gridspec_kw={"hspace": 0.3, "height_ratios": [3, 1, 1, 1]}
    )

    # Plot waveforms
    channel_names = ["Z", "N", "E"]
    for i in range(sample["X"].shape[0]):
        axs[0].plot(sample["X"][i], label=channel_names[i])
    axs[0].set_ylabel("Waveform")
    axs[0].legend()
    axs[0].set_title("Input Waveforms")

    # Plot true phase labels
    phase_names = ["P", "S", "Noise"]
    colors = ["tab:blue", "tab:green", "tab:orange"]
    for i in range(sample["y"].shape[0]):
        axs[1].plot(sample["y"][i], label=phase_names[i], color=colors[i])
    axs[1].set_ylabel("True Phase Labels")
    axs[1].legend()

    model.eval()
    with torch.no_grad():
        x = torch.tensor(sample["X"]).to(model.device).unsqueeze(0)
        x_preproc = model.annotate_batch_pre(x, {})
        phase_pred, mag_pred = model(x_preproc, return_magnitude=True)
        phase_pred = phase_pred[0].cpu().numpy()
        mag_pred = mag_pred[0, 0].cpu().numpy()  # Remove batch and channel dimensions

    # Plot predicted phase labels
    for i in range(phase_pred.shape[0]):
        axs[2].plot(
            phase_pred[i],
            label=f"Pred {phase_names[i]}",
            color=colors[i],
            linestyle="--",
        )
    axs[2].set_ylabel("Predicted Phase Labels")
    axs[2].legend()

    # Plot magnitude predictions vs true
    axs[3].plot(sample["magnitude"], label="True Magnitude", color="red", linewidth=2)
    axs[3].plot(
        mag_pred, label="Predicted Magnitude", color="blue", linestyle="--", linewidth=2
    )
    axs[3].set_ylabel("Magnitude")
    axs[3].set_xlabel("Sample Index")
    axs[3].legend()

    plt.tight_layout()
    plt.show()

    # Evaluate on all test samples with progress bar
    all_phase_preds = []
    all_phase_labels = []
    all_mag_preds = []
    all_mag_labels = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_generator)), desc="Evaluating"):
            sample = test_generator[i]
            x = torch.tensor(sample["X"]).to(model.device).unsqueeze(0)
            x_preproc = model.annotate_batch_pre(x, {})

            # Get both predictions
            phase_pred, mag_pred = model(x_preproc, return_magnitude=True)
            phase_pred = phase_pred[0].cpu().numpy()
            mag_pred = mag_pred[0, 0].cpu().numpy()

            # Store predictions and labels
            all_phase_preds.append(phase_pred)
            all_phase_labels.append(sample["y"])
            all_mag_preds.append(mag_pred)
            all_mag_labels.append(sample["magnitude"])

    # Convert to numpy arrays
    all_phase_preds = np.array(all_phase_preds)
    all_phase_labels = np.array(all_phase_labels)
    all_mag_preds = np.array(all_mag_preds)
    all_mag_labels = np.array(all_mag_labels)

    print("=" * 50)
    print("PHASE PREDICTION EVALUATION")
    print("=" * 50)

    # Phase evaluation - flatten for metrics (samples, features)
    phase_y_true = all_phase_labels.reshape(-1, all_phase_labels.shape[-1])
    phase_y_pred = all_phase_preds.reshape(-1, all_phase_preds.shape[-1])

    phase_mse = mean_squared_error(phase_y_true, phase_y_pred)
    phase_rmse = np.sqrt(phase_mse)
    phase_r2 = r2_score(phase_y_true, phase_y_pred)

    print(f"Phase MSE: {phase_mse:.6f}")
    print(f"Phase RMSE: {phase_rmse:.6f}")
    print(f"Phase R^2 Score: {phase_r2:.6f}")

    print("\n" + "=" * 50)
    print("MAGNITUDE PREDICTION EVALUATION")
    print("=" * 50)

    # Magnitude evaluation - only evaluate where we have non-zero magnitude labels
    mag_y_true_flat = all_mag_labels.flatten()
    mag_y_pred_flat = all_mag_preds.flatten()

    # Create mask for non-zero magnitude values (where we have actual targets)
    mag_mask = mag_y_true_flat != 0.0

    if mag_mask.sum() > 0:
        mag_y_true_masked = mag_y_true_flat[mag_mask]
        mag_y_pred_masked = mag_y_pred_flat[mag_mask]

        mag_mse = mean_squared_error(mag_y_true_masked, mag_y_pred_masked)
        mag_rmse = np.sqrt(mag_mse)
        mag_mae = mean_absolute_error(mag_y_true_masked, mag_y_pred_masked)
        mag_r2 = r2_score(mag_y_true_masked, mag_y_pred_masked)

        print(f"Magnitude MSE: {mag_mse:.6f}")
        print(f"Magnitude RMSE: {mag_rmse:.6f}")
        print(f"Magnitude MAE: {mag_mae:.6f}")
        print(f"Magnitude R^2 Score: {mag_r2:.6f}")
        print(f"Number of magnitude samples evaluated: {len(mag_y_true_masked):,}")

        # Additional magnitude statistics
        print(f"\nMagnitude Statistics:")
        print(
            f"True magnitude range: [{mag_y_true_masked.min():.2f}, {mag_y_true_masked.max():.2f}]"
        )
        print(
            f"Predicted magnitude range: [{mag_y_pred_masked.min():.2f}, {mag_y_pred_masked.max():.2f}]"
        )
        print(
            f"True magnitude mean ± std: {mag_y_true_masked.mean():.2f} ± {mag_y_true_masked.std():.2f}"
        )
        print(
            f"Predicted magnitude mean ± std: {mag_y_pred_masked.mean():.2f} ± {mag_y_pred_masked.std():.2f}"
        )

        # Create magnitude scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(mag_y_true_masked, mag_y_pred_masked, alpha=0.6, s=1)
        plt.plot(
            [mag_y_true_masked.min(), mag_y_true_masked.max()],
            [mag_y_true_masked.min(), mag_y_true_masked.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        plt.xlabel("True Magnitude")
        plt.ylabel("Predicted Magnitude")
        plt.title(f"Magnitude Prediction Scatter Plot (R² = {mag_r2:.3f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

        # Add some statistics to the plot
        plt.text(
            0.05,
            0.95,
            f"RMSE: {mag_rmse:.3f}\nMAE: {mag_mae:.3f}\nN: {len(mag_y_true_masked):,}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

        # Magnitude residual plot
        residuals = mag_y_pred_masked - mag_y_true_masked
        plt.figure(figsize=(10, 6))
        plt.scatter(mag_y_true_masked, residuals, alpha=0.6, s=1)
        plt.axhline(y=0, color="r", linestyle="--", label="Perfect Prediction")
        plt.xlabel("True Magnitude")
        plt.ylabel("Prediction Residuals")
        plt.title("Magnitude Prediction Residuals")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    else:
        print("No non-zero magnitude labels found in test set!")

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
