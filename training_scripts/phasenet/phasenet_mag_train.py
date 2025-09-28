import src.models.phasenet_mag as PhasenetMag
import os
from datetime import datetime
import torch
import torch.nn.functional as F

from loaders import ETHZ_loader as ETHZ

if __name__ == "__main__":
    # Load PhaseNet with magnitude prediction
    model = PhasenetMag.VariableLengthPhaseNet(
        in_samples=3001,
        phases="PSN",
        norm="std",
        default_args={"blinding": (200, 200)},
        predict_magnitude=True,
        magnitude_label="M",
    )
    model.to_preferred_device(verbose=True)

    train_generator, train_loader, _ = ETHZ.load_dataset(model, "train")
    dev_generator, dev_loader, _ = ETHZ.load_dataset(model, "dev")

    print("Data successfully loaded")

    learning_rate = 1e-3
    epochs = 50

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def combined_loss_fn(
        phase_pred,
        mag_pred,
        phase_true,
        mag_true,
        phase_weight=1.0,
        magnitude_weight=0.1,
        eps=1e-5,
    ):
        """
        Combined loss function for phase and magnitude prediction.

        Args:
            phase_pred: Predicted phase probabilities [batch, 3, samples]
            mag_pred: Predicted magnitudes [batch, 1, samples]
            phase_true: True phase labels [batch, 3, samples]
            mag_true: True magnitude labels [batch, samples]
            phase_weight: Weight for phase loss component
            magnitude_weight: Weight for magnitude loss component
            eps: Small constant for numerical stability
        """
        # Phase loss (cross-entropy)
        phase_loss = phase_true * torch.log(phase_pred + eps)
        phase_loss = (
            phase_loss.mean(-1).sum(-1).mean()
        )  # Mean over samples, sum over phases, mean over batch
        phase_loss = -phase_loss

        # Magnitude loss (MSE with masking)
        # Only compute loss where we have magnitude labels (non-zero values after P-pick)
        mag_pred_squeezed = mag_pred.squeeze(
            1
        )  # Remove channel dimension: [batch, samples]

        # Create mask for non-zero magnitude targets (where we have actual magnitude values)
        mag_mask = (mag_true != 0.0).float()

        if (
            mag_mask.sum() > 0
        ):  # Only compute magnitude loss if we have magnitude targets
            mag_loss = F.mse_loss(
                mag_pred_squeezed * mag_mask, mag_true * mag_mask, reduction="sum"
            )
            mag_loss = mag_loss / (
                mag_mask.sum() + eps
            )  # Normalize by number of magnitude samples
        else:
            mag_loss = torch.tensor(0.0, device=phase_pred.device)

        # Combined loss
        total_loss = phase_weight * phase_loss + magnitude_weight * mag_loss

        return total_loss, phase_loss, mag_loss

    def train_loop(dataloader):
        size = len(dataloader.dataset)
        model.train()

        for batch_id, batch in enumerate(dataloader):
            # Compute prediction and loss
            x = batch["X"].to(model.device)
            phase_true = batch["y"].float().to(model.device)
            mag_true = batch["magnitude"].float().to(model.device)

            x_preproc = model.annotate_batch_pre(
                x, {}
            )  # Remove mean and normalize amplitude

            # Get both phase and magnitude predictions
            phase_pred, mag_pred = model(x_preproc, return_magnitude=True)

            # Compute combined loss
            total_loss, phase_loss, mag_loss = combined_loss_fn(
                phase_pred, mag_pred, phase_true, mag_true
            )

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_id % 5 == 0:
                current = batch_id * batch["X"].shape[0]
                print(
                    f"Total loss: {total_loss.item():>7f} | "
                    f"Phase loss: {phase_loss.item():>7f} | "
                    f"Mag loss: {mag_loss.item():>7f} | "
                    f"[{current:>5d}/{size:>5d}]"
                )

    def test_loop(dataloader):
        num_batches = len(dataloader)
        test_total_loss = 0
        test_phase_loss = 0
        test_mag_loss = 0

        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(model.device)
                phase_true = batch["y"].float().to(model.device)
                mag_true = batch["magnitude"].float().to(model.device)

                x_preproc = model.annotate_batch_pre(x, {})

                # Get both phase and magnitude predictions
                phase_pred, mag_pred = model(x_preproc, return_magnitude=True)

                # Compute combined loss
                total_loss, phase_loss, mag_loss = combined_loss_fn(
                    phase_pred, mag_pred, phase_true, mag_true
                )

                test_total_loss += total_loss.item()
                test_phase_loss += phase_loss.item()
                test_mag_loss += mag_loss.item()

        model.train()

        test_total_loss /= num_batches
        test_phase_loss /= num_batches
        test_mag_loss /= num_batches

        print(
            f"Test avg - Total: {test_total_loss:>8f} | "
            f"Phase: {test_phase_loss:>8f} | "
            f"Magnitude: {test_mag_loss:>8f}\n"
        )

    # Train model with checkpoint each 5 epochs
    save_dir = "PhaseNet_Magnitude_ETHZ"
    os.makedirs(save_dir, exist_ok=True)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader)
        test_loop(dev_loader)
        if (t + 1) % 5 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"model_epoch_{t+1}_{timestamp}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_final_{timestamp}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete, model saved to {save_path}")
