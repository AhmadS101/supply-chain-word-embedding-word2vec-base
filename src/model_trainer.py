import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Get absolute path to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)


from src.build_dataset import dataset_loader
from src.cbow_model import model, loss_fn, optimizer


# Training Neural Network Model
epochs = 200


epochs_counter = []
loss_values = []
loss_stds = []


for epoch in range(epochs):
    epoch_losses = []

    for target, context in dataset_loader:
        optimizer.zero_grad()

        output = model(context)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    epoch_mean_loss = np.mean(epoch_losses)
    epoch_std = np.std(epoch_losses)

    epochs_counter.append(epoch)
    loss_values.append(epoch_mean_loss)
    loss_stds.append(epoch_std)

    print(f"Epoch {epoch:3d} | Loss: {epoch_mean_loss:.4f} ± {epoch_std:.4f}")


torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_values[-1],
    },
    "../models/checkpoints.pth",
)

plt.figure(figsize=(10, 5))
plt.plot(epochs_counter, loss_values, label="Loss", color="green")
plt.fill_between(
    epochs_counter,
    np.array(loss_values) - np.array(loss_stds),
    np.array(loss_values) + np.array(loss_stds),
    alpha=0.2,
    color="red",
)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
