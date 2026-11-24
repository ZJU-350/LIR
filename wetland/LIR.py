import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

import torch.nn.functional as F


class BrayCurtisLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_pred, x_true):
        diff = torch.abs(x_pred - x_true)
        numerator = diff.sum(dim=-1) # Sum by sample
        sum_add = x_pred + x_true
        denominator = sum_add.sum(dim=-1) + self.epsilon
        bc_dissimilarity = numerator / denominator
        return bc_dissimilarity.mean()



class LIR_Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        raw_abundance = s['abundance']

        indices = np.nonzero(raw_abundance)[0]
        values = raw_abundance[indices]

        indices_tensor = torch.LongTensor(indices)
        values_tensor = torch.FloatTensor(values)

        env_1 = torch.LongTensor([s['env_1']])
        env_2 = torch.LongTensor([s['env_2']])

        return indices_tensor, values_tensor, env_1, env_2


class LIR_Collate:
    def __init__(self, padding_value: int):
        self.padding_value = padding_value

    def __call__(self, batch):
        indices_list, values_list, env_list_1, env_list_2 = zip(*batch)
        padded_ids = pad_sequence(indices_list, batch_first=True, padding_value=self.padding_value)
        padded_abundances = pad_sequence(values_list, batch_first=True, padding_value=0.0)
        env_batch_1 = torch.stack(env_list_1).view(-1)
        env_batch_2 = torch.stack(env_list_2).view(-1)
        return padded_ids, padded_abundances, env_batch_1, env_batch_2


class LIR_Model(nn.Module):
    def __init__(self, num_total_species, d_model, dropout):
        super().__init__()
        self.num_total_species = num_total_species
        self.d_model = d_model
        self.padding_idx = num_total_species
        self.max_r = 3.0
        self.max_interaction = 5.0

        # Embedding: num_total_species + 1 (for padding)
        self.emb_query = nn.Embedding(num_total_species + 1, d_model, padding_idx=self.padding_idx)
        self.emb_key = nn.Embedding(num_total_species + 1, d_model, padding_idx=self.padding_idx)
        self.emb_r = nn.Embedding(num_total_species + 1, 1, padding_idx=self.padding_idx)
        self.diag_emb = nn.Embedding(num_total_species + 1, 1, padding_idx=self.padding_idx)

        self.env_emb_1 = nn.Embedding(6, d_model)
        self.env_emb_2 = nn.Embedding(2, d_model)

        output_dim = 2 * d_model + 2
        self.adapter = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model * 2, output_dim)
        )

        self.feat_dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.emb_query.weight, std=0.01)
        nn.init.normal_(self.emb_key.weight, std=0.01)
        nn.init.normal_(self.emb_r.weight, mean=0.5, std=0.1)
        nn.init.normal_(self.diag_emb.weight, mean=1, std=0.1)
        nn.init.normal_(self.env_emb_1.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.env_emb_2.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)


    def get_dynamics_params(self, species_ids, sample_env_1, sample_env_2):
        Q_local = self.emb_query(species_ids)
        K_local = self.emb_key(species_ids)
        r_local = self.emb_r(species_ids).squeeze(-1)
        diag_local = self.diag_emb(species_ids).squeeze(-1)

        batch_size, num_species = species_ids.shape
        env_emb_1 = self.env_emb_1(sample_env_1)
        env_emb_2 = self.env_emb_2(sample_env_2)
        env_context = env_emb_1 + env_emb_2

        env_expanded = env_context.unsqueeze(1).expand(-1, num_species, -1)
        specific_context = torch.cat([Q_local, env_expanded], dim=-1)  # [B, L, 2*d]

        all_params = self.adapter(specific_context)

        scale, shift, r_bias, diag_scale_raw = torch.split(
            all_params,
            [self.d_model, self.d_model, 1, 1],
            dim=-1
        )

        r_bias = r_bias.squeeze(-1)  # [B, L, 1] -> [B, L]
        r_final = r_local + r_bias


        diag_base = -F.softplus(diag_local)
        diag_scale = F.softplus(diag_scale_raw).squeeze(-1)  # > 0
        diag_final = diag_base * diag_scale  # < 0

        Q_final = Q_local * (1 + scale) + shift
        K_final = K_local * (1 + scale) + shift

        # Dropout
        Q_final = self.feat_dropout(Q_final)
        K_final = self.feat_dropout(K_final)

        return r_final, Q_final, K_final, diag_final

    def _compute_interaction(self, Q, K, diag_vals):
        scaling = self.d_model ** 0.5
        interaction = torch.bmm(Q, K.transpose(1, 2)) / scaling
        interaction = self.max_interaction * torch.tanh(interaction / self.max_interaction)

        eye = torch.eye(Q.shape[1], device=Q.device).unsqueeze(0)
        interaction = interaction * (1.0 - eye) + (diag_vals.unsqueeze(1) * eye)
        return interaction

    def forward(self, species_ids, initial_abundances, sample_env_1, sample_env_2):
        device = species_ids.device
        mask = (initial_abundances > 0).float()

        r, Q, K, diag_vals = self.get_dynamics_params(species_ids, sample_env_1, sample_env_2)
        r = self.max_r * torch.sigmoid(r) * mask  # r > 0 and mask padding

        interaction = self._compute_interaction(Q, K, diag_vals)

        mask_mat = mask.unsqueeze(1) * mask.unsqueeze(2)
        interaction = interaction * mask_mat

        presence_mask = (initial_abundances > 0).float()
        count_species = presence_mask.sum(dim=-1, keepdim=True) + 1e-12
        x0 = presence_mask / count_species

        def glv_func(t, x):
            x = x * mask
            x_safe = torch.abs(x)
            x_safe = torch.clamp(x_safe, max=1e5)  # Prevent forward explosion
            inter_effect = torch.bmm(interaction, x_safe.unsqueeze(-1)).squeeze(-1)
            dx = x_safe * (r + inter_effect)
            return dx * mask

        t_span = torch.tensor([0.0, 1], device=device)
        x_final = odeint(glv_func, x0, t_span, method='implicit_adams', rtol=1e-3, atol=1e-4)[-1]
        x_final = torch.relu(x_final) * mask.float()
        pred_abundance = x_final / (x_final.sum(dim=-1, keepdim=True) + 1e-12)

        return pred_abundance, interaction

    @torch.no_grad()
    def get_interaction_matrix(self, env_1_id, env_2_id) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            device = self.emb_query.weight.device
            env_1_tensor = torch.tensor([env_1_id], dtype=torch.long, device=device)
            env_2_tensor = torch.tensor([env_2_id], dtype=torch.long, device=device)
            all_species_indices = torch.arange(0, self.num_total_species, device=device).unsqueeze(0)
            _, Q, K, diag_vals = self.get_dynamics_params(all_species_indices, env_1_tensor, env_2_tensor)
            interaction = self._compute_interaction(Q, K, diag_vals)

            matrix = interaction.squeeze(0).cpu().numpy()

            return matrix
        finally:
            self.train(was_training)



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_l1_loss = 0
    total_pos_loss = 0


    for padded_ids, padded_abundances, env_1, env_2 in dataloader:
        padded_ids = padded_ids.to(device)
        padded_abundances = padded_abundances.to(device)
        env_1 = env_1.to(device)
        env_2 = env_2.to(device)

        pred, interaction_matrix = model(padded_ids, padded_abundances, env_1, env_2)

        loss_main = criterion(pred, padded_abundances)
        loss_l1 = torch.abs(interaction_matrix).mean()
        loss_pos = torch.relu(interaction_matrix).mean()

        l1_lambda = 1e-5
        pos_lambda = 0.5
        loss = loss_main + (l1_lambda * loss_l1) + (pos_lambda * loss_pos)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_main_loss += loss_main.item()
        total_l1_loss += loss_l1.item()
        total_pos_loss += loss_pos.item()

    return total_loss / len(dataloader)



def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for padded_ids, padded_abundances, env_1, env_2 in dataloader:
            padded_ids = padded_ids.to(device)
            padded_abundances = padded_abundances.to(device)
            env_1 = env_1.to(device)
            env_2 = env_2.to(device)

            pred, _ = model(padded_ids, padded_abundances, env_1, env_2)
            loss = criterion(pred, padded_abundances)
            total_loss += loss.item()

    return total_loss / len(dataloader)




