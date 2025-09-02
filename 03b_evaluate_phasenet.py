import seisbench.models as sbm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import torch
from tqdm import tqdm
import numpy as np

import loaders.ETHZ_loader as ETHZ


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PhaseNet model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint file"
    )
    args = parser.parse_args()

    # Load standard Phasenet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    model.to_preferred_device(verbose=True)

    # Load weights from specified checkpoint
    state_dict = torch.load(args.model_path, map_location=model.device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {args.model_path}")

    test_generator, _, _ = ETHZ.load_dataset(model, "test")

    # Visual check: plot a random test sample and prediction
    sample = test_generator[np.random.randint(len(test_generator))]
    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(
        3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]}
    )
    axs[0].plot(sample["X"].T)
    axs[1].plot(sample["y"].T)

    model.eval()
    with torch.no_grad():
        x = torch.tensor(sample["X"]).to(model.device).unsqueeze(0)
        x_preproc = model.annotate_batch_pre(x, {})
        pred = model(x_preproc)[0].cpu().numpy()
    axs[2].plot(pred.T)
    plt.show()

    # Evaluate on all test samples with progress bar
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i in tqdm(range(len(test_generator)), desc="Evaluating"):
            sample = test_generator[i]
            x = torch.tensor(sample["X"]).to(model.device).unsqueeze(0)
            x_preproc = model.annotate_batch_pre(x, {})
            pred = model(x_preproc)[0].cpu().numpy()
            label = sample["y"]
            all_preds.append(pred)
            all_labels.append(label)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Flatten for metrics (samples, features)
    y_true = all_labels.reshape(-1, all_labels.shape[-1])
    y_pred = all_preds.reshape(-1, all_preds.shape[-1])

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test R^2 Score: {r2:.6f}")
