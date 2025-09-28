# train_amag.py
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader

import seisbench.models as sbm  # still useful for utils
import loaders.ETHZ_loader as ETHZ_loader  # your existing loader

from models.amag_pytorch_v1 import AMAGMag, make_optimizer, train_step, eval_step

if __name__ == "__main__":
    # 1) Build model compatible with SeisBench model helpers
    model = AMAGMag(
        in_channels=3, base_filters=32, depth=4, kernel_size=5, use_eca=True, norm="std"
    )
    model.to_preferred_device(verbose=True)
    device = next(model.parameters()).device

    # 2) Load datasets/loaders exactly like your PhaseNet training
    train_gen, train_loader, _ = ETHZ_loader.load_dataset(model, split="train")
    dev_gen, dev_loader, _ = ETHZ_loader.load_dataset(model, split="dev")

    # 3) Optimizer
    optimizer = make_optimizer(model, lr=1e-3, weight_decay=1e-4)

    # 4) Training loop
    epochs = 20
    save_dir = "checkpoints_amag"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            running += train_step(batch, model, optimizer, device)
        train_loss = running / max(1, len(train_loader))

        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                running += eval_step(batch, model, device)
        dev_loss = running / max(1, len(dev_loader))

        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | dev {dev_loss:.4f}")

        if epoch % 5 == 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt = os.path.join(save_dir, f"amag_epoch{epoch}_{ts}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved {ckpt}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = os.path.join(save_dir, f"amag_final_{ts}.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Done. Saved {ckpt}")
