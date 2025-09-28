from models import eqt_mag as EQT
import os
from datetime import datetime
import torch
import torch.nn.functional as F

from loaders import ETHZ_loader as ETHZ

if __name__ == "__main__":
    # Load EQTransformer with magnitude prediction
    model = EQT.EQTransformer(
        in_channels=3,
        in_samples=6000,
        classes=2,
        phases="PS",
        norm="std",
        default_args={"blinding": (500, 500)},
        predict_magnitude=True,
        magnitude_label="M",
    )
    model.to_preferred_device(verbose=True)

    train_generator, train_loader, _ = ETHZ.load_dataset(model, "train")
    dev_generator, dev_loader, _ = ETHZ.load_dataset(model, "dev")

    print("Data successfully loaded")

    learning_rate = 1e-3  # Lower learning rate for EQTransformer
    epochs = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def combined_loss_fn(
        outputs,
        phase_true,
        mag_true,
        detection_weight=1.0,
        phase_weight=1.0,
        magnitude_weight=0.1,
        eps=1e-5,
    ):
        """
        Combined loss function for EQTransformer with magnitude prediction.

        Args:
            outputs: Tuple of (detection, P_phase, S_phase, magnitude) predictions
            phase_true: True phase labels [batch, 3, samples] (P, S, N)
            mag_true: True magnitude labels [batch, samples]
            detection_weight: Weight for detection loss
            phase_weight: Weight for phase loss component
            magnitude_weight: Weight for magnitude loss component
            eps: Small constant for numerical stability
        """
        if len(outputs) == 4:  # With magnitude
            detection_pred, p_pred, s_pred, mag_pred = outputs
        else:  # Without magnitude
            detection_pred, p_pred, s_pred = outputs
            mag_pred = None

        # Detection loss (binary cross-entropy with logits)
        # Create detection target: 1 where P or S phases are present, 0 otherwise
        detection_true = torch.clamp(
            phase_true[:, 0] + phase_true[:, 1], 0, 1
        )  # P + S phases
        detection_loss = F.binary_cross_entropy_with_logits(
            detection_pred, detection_true
        )

        # Phase losses (binary cross-entropy with logits for each phase)
        p_loss = F.binary_cross_entropy_with_logits(p_pred, phase_true[:, 0])  # P phase
        s_loss = F.binary_cross_entropy_with_logits(s_pred, phase_true[:, 1])  # S phase
        phase_loss = (p_loss + s_loss) / 2

        # Magnitude loss (MSE with masking)
        if mag_pred is not None:
            # Create mask for non-zero magnitude targets
            mag_mask = (mag_true != 0.0).float()

            if mag_mask.sum() > 0:
                mag_loss = F.mse_loss(
                    mag_pred * mag_mask, mag_true * mag_mask, reduction="sum"
                )
                mag_loss = mag_loss / (mag_mask.sum() + eps)
            else:
                mag_loss = torch.tensor(0.0, device=detection_pred.device)
        else:
            mag_loss = torch.tensor(0.0, device=detection_pred.device)

        # Combined loss
        total_loss = (
            detection_weight * detection_loss
            + phase_weight * phase_loss
            + magnitude_weight * mag_loss
        )

        return total_loss, detection_loss, phase_loss, mag_loss

    def train_loop(dataloader):
        size = len(dataloader.dataset)
        model.train()

        for batch_id, batch in enumerate(dataloader):
            # Compute prediction and loss
            x = batch["X"].to(model.device)
            phase_true = batch["y"].float().to(model.device)
            mag_true = batch["magnitude"].float().to(model.device)

            x_preproc = model.annotate_batch_pre(x, {})

            # Get predictions (detection, P, S, magnitude)
            outputs = model(x_preproc, return_magnitude=True, logits=True)

            # Compute combined loss
            total_loss, detection_loss, phase_loss, mag_loss = combined_loss_fn(
                outputs, phase_true, mag_true
            )

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_id % 5 == 0:
                current = batch_id * batch["X"].shape[0]
                print(
                    f"Total: {total_loss.item():>7f} | "
                    f"Det: {detection_loss.item():>6f} | "
                    f"Phase: {phase_loss.item():>6f} | "
                    f"Mag: {mag_loss.item():>6f} | "
                    f"[{current:>5d}/{size:>5d}]"
                )

    def test_loop(dataloader):
        num_batches = len(dataloader)
        test_total_loss = 0
        test_detection_loss = 0
        test_phase_loss = 0
        test_mag_loss = 0

        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(model.device)
                phase_true = batch["y"].float().to(model.device)
                mag_true = batch["magnitude"].float().to(model.device)

                x_preproc = model.annotate_batch_pre(x, {})

                # Get predictions
                outputs = model(x_preproc, return_magnitude=True, logits=True)

                # Compute combined loss
                total_loss, detection_loss, phase_loss, mag_loss = combined_loss_fn(
                    outputs, phase_true, mag_true
                )

                test_total_loss += total_loss.item()
                test_detection_loss += detection_loss.item()
                test_phase_loss += phase_loss.item()
                test_mag_loss += mag_loss.item()

        model.train()

        test_total_loss /= num_batches
        test_detection_loss /= num_batches
        test_phase_loss /= num_batches
        test_mag_loss /= num_batches

        print(
            f"Test avg - Total: {test_total_loss:>8f} | "
            f"Detection: {test_detection_loss:>8f} | "
            f"Phase: {test_phase_loss:>8f} | "
            f"Magnitude: {test_mag_loss:>8f}\n"
        )

    # Train model with checkpoint each 5 epochs
    save_dir = "trained_weights/EQTransformer_Magnitude_ETHZ"
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
