import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math
import os
from torch.utils.data import Dataset, DataLoader
from LIR import train_one_epoch, LIR_Model, BrayCurtisLoss, LIR_Collate, LIR_Dataset
import random


def loss_curve(loss_data, boot_idx):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14

    df = pd.DataFrame(loss_data, columns=["epoch", "train"])
    final_loss = df["train"].values[-1]
    plt.figure(figsize=(8, 6), constrained_layout=True)

    plt.plot(df["epoch"], df["train"], label="Train Loss", color="blue", alpha=0.7)
    plt.ylim(bottom=-0.05, top=1)
    plt.xlabel("number of training epochs", fontsize=14)
    plt.ylabel("prediction error", fontsize=14)
    plt.legend()
    plt.title(f"final loss: {final_loss:.6f} ", fontsize=14)
    plt.grid(True)

    plt.savefig(f"bootstrap/model_{boot_idx}.png", dpi=600)
    plt.close()



if __name__ == "__main__":
    df_abundance = pd.read_csv("dataset/filtered_abundance.csv", index_col=0)
    df_abundance.index = df_abundance.index.astype(str)

    df_meta = pd.read_csv("dataset/filtered_meta.csv")
    df_meta["Name"] = df_meta["Name"].astype(str)
    # Align metadata according to the row order of the abundance table
    df_meta = df_meta.set_index("Name").loc[df_abundance.index]

    final_species = df_abundance.columns.tolist()
    n_species = len(final_species)
    print(f"Total Species: {n_species}")

    abund_matrix = df_abundance.values.astype(np.float32)
    env_1_arr = df_meta["env_1"].values.astype(np.int64)
    env_2_arr = df_meta["env_2"].values.astype(np.int64)

    samples = [
        {
            "name": sample_name,
            "abundance": abund_matrix[i],
            "env_1": int(env_1_arr[i]),
            "env_2": int(env_2_arr[i]),
        }
        for i, sample_name in enumerate(df_abundance.index)
    ]
    print(f"Total samples: {len(samples)}")

    n_bootstraps = 1000 # bootstrap epochs
    n_epochs = 100  # train epochs
    save_dir = "bootstrap/"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # ================= Bootstrap =================
    for boot_idx in range(n_bootstraps):
        save_path = os.path.join(save_dir, f"model_{boot_idx}.pth")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            continue

        seed = 991011 + boot_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        boot_samples = random.choices(samples, k=len(samples))

        train_dataset = LIR_Dataset(boot_samples)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=LIR_Collate(padding_value=n_species))

        model = LIR_Model(num_total_species=n_species, d_model=32, dropout=0.3).to(device)

        criterion = BrayCurtisLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=1, eta_min=1e-4
        )

        model.train()
        best_loss_run = math.inf
        loss_data = []
        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            best_loss_run = min(train_loss, best_loss_run)
            loss_data.append([epoch, train_loss])
            if (epoch + 1) % 50 == 0 or (epoch + 1) == n_epochs:
                print(f"  > Epoch {epoch + 1:03d} | Loss: {train_loss:.6f} (Best: {best_loss_run:.6f})")
        boot_abund = np.array([s['abundance'] for s in boot_samples])
        save_dict = {
            'model_state': model.state_dict(),
            'present_species': (np.nonzero(boot_abund.sum(axis=0))[0]).tolist()
        }
        tmp_path = save_path + ".tmp"
        torch.save(save_dict, tmp_path)
        os.replace(tmp_path, save_path)
        loss_curve(loss_data, boot_idx)
        print(f" > Model saved to: {save_path}")

    print("\nAll Bootstrap training rounds finished!")