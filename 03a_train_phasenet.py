import seisbench.models as sbm
import os
from datetime import datetime
import torch

import loaders.ETHZ_loader as ETHZ

if __name__ == "__main__":
    # Load standard Phasenet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    model.to_preferred_device(verbose=True)

    train_generator, train_loader, _ = ETHZ.load_dataset(model, "train")
    dev_generator, dev_loader, _ = ETHZ.load_dataset(model, "dev")

    print("Data successfully loaded")

    learning_rate = 1e-2
    epochs = 5

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss_fn(y_pred, y_true, eps=1e-5):
        # vector cross entropy loss
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(
            -1
        )  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h

    def train_loop(dataloader):
        size = len(dataloader.dataset)
        for batch_id, batch in enumerate(dataloader):
            # Compute prediction and loss
            x = batch["X"].to(model.device)
            x_preproc = model.annotate_batch_pre(
                x, {}
            )  # Remove mean and normalize amplitude
            pred = model(x_preproc)
            loss = loss_fn(pred, batch["y"].float().to(model.device))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 5 == 0:
                loss, current = loss.item(), batch_id * batch["X"].shape[0]
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader):
        num_batches = len(dataloader)
        test_loss = 0

        model.eval()  # close the model for evaluation

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(model.device)
                x_preproc = model.annotate_batch_pre(
                    x, {}
                )  # Remove mean and normalize amplitude
                pred = model(x_preproc)
                test_loss += loss_fn(pred, batch["y"].float().to(model.device)).item()

        model.train()  # re-open model for training stage

        test_loss /= num_batches
        print(f"Test avg loss: {test_loss:>8f} \n")

    # Train model with checkpoint each 5 epochs
    save_dir = "PhaseNet_ETHZ"
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
