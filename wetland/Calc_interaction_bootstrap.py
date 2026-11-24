import os
import torch
import numpy as np
import pandas as pd
from LIR import LIR_Model


def load_and_extract_submatrix(model, model_path, target_species_indices, device, env_1_id, env_2_id):
    if not os.path.exists(model_path):
        return None, None

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'present_env_1' in checkpoint and env_1_id not in checkpoint['present_env_1']:
            return None, None
        if 'present_env_2' in checkpoint and env_2_id not in checkpoint['present_env_2']:
            return None, None

        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        boot_present_set = set(checkpoint.get('present_species', target_species_indices))
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None, None

    # 仅保留当前生境活跃且在当轮 Bootstrap 出现过的物种（长度 <= 1357，避免 16923x16923 内存爆炸）
    valid_global_idxs = [gid for gid in target_species_indices if gid in boot_present_set]
    if not valid_global_idxs:
        return None, None

    model.eval()
    with torch.no_grad():
        species_tensor = torch.tensor(valid_global_idxs, dtype=torch.long, device=device).unsqueeze(0)
        env_1_tensor = torch.tensor([env_1_id], dtype=torch.long, device=device)
        env_2_tensor = torch.tensor([env_2_id], dtype=torch.long, device=device)

        _, Q, K, diag_vals = model.get_dynamics_params(species_tensor, env_1_tensor, env_2_tensor)
        interaction = model._compute_interaction(Q, K, diag_vals)
        sub_mat = interaction.squeeze(0).cpu().numpy()

    return sub_mat, valid_global_idxs


if __name__ == '__main__':
    n_bootstraps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_abundance = pd.read_csv("dataset/filtered_abundance.csv", index_col=0)
    all_species_names = df_abundance.columns.tolist()
    n_species = len(all_species_names)
    print(f"Total Species: {n_species} | Total Samples: {len(df_abundance)}")

    df_meta = pd.read_csv("dataset/filtered_meta.csv")
    meta_aligned = (
        df_meta.assign(Name=df_meta["Name"].astype(str))
        .set_index("Name")
        .loc[df_abundance.index.astype(str)]
    )

    tasks = list(
        df_meta[["env_1", "env_2"]]
        .drop_duplicates()
        .sort_values(["env_1", "env_2"])
        .itertuples(index=False, name=None)
    )

    model = LIR_Model(num_total_species=n_species, d_model=32, dropout=0.1).to(device)
    model.eval()

    for env_1, env_2 in tasks:
        print(f"\n=== Processing: env_1={env_1}, env_2={env_2} ===")

        env_mask = ((meta_aligned["env_1"] == env_1) & (meta_aligned["env_2"] == env_2)).values
        sub_abund = df_abundance.values[env_mask]

        if len(sub_abund) == 0:
            print("   > No samples found. Skipped.")
            continue

        active_indices = np.nonzero(sub_abund.sum(axis=0))[0].tolist()
        n_active = len(active_indices)
        print(f"   > Samples: {len(sub_abund)} | Active species: {n_active}")
        if n_active == 0:
            continue

        active_species_names = [all_species_names[idx] for idx in active_indices]
        global_to_local = {gid: idx for idx, gid in enumerate(active_indices)}

        presence_sub = (sub_abund[:, active_indices] > 0).astype(np.int32)
        never_cooccur_mask = (presence_sub.T @ presence_sub) == 0

        sum_matrix = np.zeros((n_active, n_active), dtype=np.float64)
        valid_counts = np.zeros((n_active, n_active), dtype=np.int32)
        count_gt_0 = np.zeros((n_active, n_active), dtype=np.int32)
        count_lt_0 = np.zeros((n_active, n_active), dtype=np.int32)

        processed_count = 0

        for i in range(n_bootstraps):
            path = os.path.join("bootstrap", f"model_{i}.pth")
            sub_mat, valid_global_idxs = load_and_extract_submatrix(
                model=model,
                model_path=path,
                target_species_indices=active_indices,
                device=device,
                env_1_id=env_1,
                env_2_id=env_2
            )
            if sub_mat is None:
                continue

            local_indices = [global_to_local[gid] for gid in valid_global_idxs]
            local_ix = np.ix_(local_indices, local_indices)

            sum_matrix[local_ix] += sub_mat
            valid_counts[local_ix] += 1
            count_gt_0[local_ix] += (sub_mat > 0).astype(np.int32)
            count_lt_0[local_ix] += (sub_mat < 0).astype(np.int32)

            processed_count += 1

        if processed_count == 0:
            print(f"   > Warning: No valid models loaded for ({env_1}, {env_2}). Skipped.")
            continue

        valid_counts_safe = np.maximum(valid_counts, 1)
        mean_mat = sum_matrix / valid_counts_safe

        prop_gt = count_gt_0 / valid_counts_safe
        prop_lt = count_lt_0 / valid_counts_safe
        p_mat = 2.0 * np.minimum(prop_gt, prop_lt)

        min_possible_p = 1.0 / valid_counts_safe
        p_mat = np.maximum(p_mat, min_possible_p)

        p_mat[never_cooccur_mask] = 1.0
        mean_mat[never_cooccur_mask] = 0.0

        np.fill_diagonal(mean_mat, 0.0)
        np.fill_diagonal(p_mat, 1.0)

        out_dir = "result/interaction"
        os.makedirs(out_dir, exist_ok=True)

        df_interaction = pd.DataFrame(mean_mat, index=active_species_names, columns=active_species_names)
        df_interaction.to_csv(f"{out_dir}/{env_1}_{env_2}_interaction_{processed_count}.csv")

        df_pvalues = pd.DataFrame(p_mat, index=active_species_names, columns=active_species_names)
        df_pvalues.to_csv(f"{out_dir}/{env_1}_{env_2}_p_{processed_count}.csv")

        count_pos_sig = int(((df_interaction > 0) & (df_pvalues < 0.01)).sum().sum())
        count_neg_sig = int(((df_interaction < 0) & (df_pvalues < 0.01)).sum().sum())
        print(f"[{env_1}_{env_2}]  (Positive & P<0.01): {count_pos_sig}")
        print(f"[{env_1}_{env_2}]  (Negative & P<0.01): {count_neg_sig}")

        df_counts = pd.DataFrame(valid_counts, index=active_species_names, columns=active_species_names)
        df_details = pd.concat([df_interaction.stack(), df_pvalues.stack(), df_counts.stack()], axis=1)
        df_details.columns = ['Weight', 'P_value', 'Stability']
        df_details.reset_index(inplace=True)
        df_details.rename(columns={'level_0': 'Source', 'level_1': 'Target'}, inplace=True)

        df_details = df_details[
            (df_details['Source'] != df_details['Target']) &
            (df_details['P_value'] < 0.05) &
            (df_details['Weight'] != 0)
        ].copy()

        if not df_details.empty:
            df_details['Abs_Weight'] = df_details['Weight'].abs()
            df_details = df_details.sort_values(
                by=['P_value', 'Abs_Weight'], ascending=[True, False]
            ).drop(columns=['Abs_Weight'])

            save_path = f"{out_dir}/{env_1}_{env_2}_interaction_details_{processed_count}.csv"
            df_details.to_csv(save_path, index=False)
            print(f"   > Details saved to {save_path} ({len(df_details)} edges)")
        else:
            print("   > No significant edges found. Skipped saving details.")

    print("\nAll tasks completed.")