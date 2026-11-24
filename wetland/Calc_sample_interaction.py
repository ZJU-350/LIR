import os
import glob
import itertools
import numpy as np
import pandas as pd


def sample_interaction_statistics():
    # 1. 直接读取预处理并对齐后的丰度表与元数据
    df_abundance = pd.read_csv("dataset/filtered_abundance.csv", index_col=0)
    df_abundance.index = df_abundance.index.astype(str)

    df_meta = pd.read_csv("dataset/filtered_meta.csv")
    df_meta["Name"] = df_meta["Name"].astype(str)
    df_meta = df_meta.set_index("Name").loc[df_abundance.index]

    final_species = df_abundance.columns.tolist()
    n_species = len(final_species)
    print(f"Total Species: {n_species} | Total Preprocessed Samples: {len(df_abundance)}")

    # 加载物种分类学注释表
    df_taxa = pd.read_csv("dataset/tree_species.csv")
    TAX_LEVELS = ['Phylum', 'Class', 'Order', 'Family', 'Genus']
    tax_maps = {
        lvl: dict(zip(df_taxa['Species'], df_taxa[lvl].fillna("Unknown")))
        for lvl in TAX_LEVELS if lvl in df_taxa.columns
    }

    abund_matrix = df_abundance.values
    all_species_arr = np.array(final_species)
    env_1_arr = df_meta['env_1'].values.astype(int)
    env_2_arr = df_meta['env_2'].values.astype(int)
    type_arr = df_meta['Type'].astype(str).str.strip().str.lower().values

    samples = []
    for i, sample_name in enumerate(df_abundance.index):
        e1 = int(env_1_arr[i])
        e2 = int(env_2_arr[i])
        samples.append({
            'name': sample_name,
            'env_1': e1,
            'env_2': e2,
            'env_name': type_arr[i],
            'medium_name': 'water' if e2 == 1 else 'soil',
            'species': all_species_arr[abund_matrix[i] > 0].tolist()
        })


    print("\nPre-loading interaction edge lists...")
    interaction_cache = {}
    unique_contexts = sorted(set((s['env_1'], s['env_2']) for s in samples))

    for env_1, env_2 in unique_contexts:

        pattern = f"result/interaction/{env_1}_{env_2}_interaction_details_*.csv"
        matched_files = sorted(glob.glob(pattern))

        if not matched_files:
            print(f"Warning: No interaction details file found for ({env_1}, {env_2}): {pattern}")
            interaction_cache[(env_1, env_2)] = {}
            continue

        csv_path = matched_files[-1]
        try:
            df_edges = pd.read_csv(csv_path)
            edge_map = dict(zip(zip(df_edges['Source'], df_edges['Target']), df_edges['Weight']))
            interaction_cache[(env_1, env_2)] = edge_map
            print(f"   Loaded ({env_1}, {env_2}) from {csv_path}: {len(edge_map)} edges")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            interaction_cache[(env_1, env_2)] = {}

    # 3. 遍历样本计算互作统计量
    print("\nCalculating sample-wise interactions...")
    stats_results = []

    for sample in samples:
        sample_name = sample['name']
        e1 = sample['env_1']
        e2 = sample['env_2']
        present_species = sample['species']

        n_present_species = len(present_species)
        max_potential_edges = n_present_species * (n_present_species - 1)
        edge_map = interaction_cache.get((e1, e2), {})

        count_total_edges = 0
        count_pos_edges = 0
        count_neg_edges = 0

        type_counts = {
            'Mutualism': 0,      # (+, +)
            'Exploitation': 0,   # (+, -) or (-, +)
            'Competition': 0,    # (-, -)
            'Commensalism': 0,   # (+, 0) or (0, +)
            'Amensalism': 0      # (-, 0) or (0, -)
        }

        level_counts = {
            lvl: {'intra_total': 0, 'intra_pos': 0, 'inter_total': 0, 'inter_pos': 0}
            for lvl in tax_maps
        }

        for sp_a, sp_b in itertools.combinations(present_species, 2):
            w_ab = edge_map.get((sp_a, sp_b), 0.0)
            w_ba = edge_map.get((sp_b, sp_a), 0.0)

            if w_ab != 0:
                count_total_edges += 1
                if w_ab > 0:
                    count_pos_edges += 1
                else:
                    count_neg_edges += 1

            if w_ba != 0:
                count_total_edges += 1
                if w_ba > 0:
                    count_pos_edges += 1
                else:
                    count_neg_edges += 1

            if w_ab == 0 and w_ba == 0:
                continue

            if w_ab > 0 and w_ba > 0:
                type_counts['Mutualism'] += 1
            elif w_ab < 0 and w_ba < 0:
                type_counts['Competition'] += 1
            elif (w_ab > 0 and w_ba < 0) or (w_ab < 0 and w_ba > 0):
                type_counts['Exploitation'] += 1
            elif (w_ab > 0 and w_ba == 0) or (w_ab == 0 and w_ba > 0):
                type_counts['Commensalism'] += 1
            elif (w_ab < 0 and w_ba == 0) or (w_ab == 0 and w_ba < 0):
                type_counts['Amensalism'] += 1

            for lvl, s2t in tax_maps.items():
                tax_a = s2t.get(sp_a, "Unknown")
                tax_b = s2t.get(sp_b, "Unknown")
                if tax_a == "Unknown" or tax_b == "Unknown":
                    continue

                is_same = (tax_a == tax_b)
                for w in (w_ab, w_ba):
                    if w == 0:
                        continue
                    if is_same:
                        level_counts[lvl]['intra_total'] += 1
                        if w > 0:
                            level_counts[lvl]['intra_pos'] += 1
                    else:
                        level_counts[lvl]['inter_total'] += 1
                        if w > 0:
                            level_counts[lvl]['inter_pos'] += 1

        row = {
            'Sample': sample_name,
            'Environment': sample['env_name'],
            'Medium': sample['medium_name'],
            'N_Species': n_present_species,
            'Max_Potential_Edges': max_potential_edges,
            'Total_Interactions': count_total_edges,
            'Positive_Interactions': count_pos_edges,
            'Negative_Interactions': count_neg_edges,
            'Mutualism': type_counts['Mutualism'],
            'Exploitation': type_counts['Exploitation'],
            'Competition': type_counts['Competition'],
            'Commensalism': type_counts['Commensalism'],
            'Amensalism': type_counts['Amensalism'],
        }

        for lvl in tax_maps:
            c = level_counts[lvl]
            row[f'Intra_{lvl}'] = c['intra_pos'] / c['intra_total'] if c['intra_total'] > 0 else np.nan
            row[f'Inter_{lvl}'] = c['inter_pos'] / c['inter_total'] if c['inter_total'] > 0 else np.nan

        stats_results.append(row)

    # 4. 导出统计结果
    if stats_results:
        df_result = pd.DataFrame(stats_results)
        cols = [
            'Sample', 'Environment', 'Medium', 'N_Species', 'Max_Potential_Edges',
            'Total_Interactions', 'Positive_Interactions', 'Negative_Interactions',
            'Mutualism', 'Exploitation', 'Competition', 'Commensalism', 'Amensalism',
        ] + [f'{p}_{lvl}' for lvl in tax_maps for p in ('Intra', 'Inter')]
        df_result = df_result[cols]


        output_filename = "result/sample_interaction_statistics.csv"
        df_result.to_csv(output_filename, index=False)
        print(f"\nSaved statistics to {output_filename}")
        print(df_result.head())
    else:
        print("No results generated.")


if __name__ == "__main__":
    sample_interaction_statistics()