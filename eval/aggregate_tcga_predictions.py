"""
Aggregate per-ROI TCGA predictions (from classify_tcga.py) against TIGER GT,
into the summary CSVs eval/compute_metrics.py reads.

  - GT from ASAP XML annotations (WSI-level polygons) instead of COCO JSON
  - Stardist .mat inst_map (already instance-labeled, no label() needed)
  - 4 classifiers: lizard_baseline, panoptils_baseline, stage1_context, stage2_loki_ot
  - PanopTILs uses class 3 (TILs) as lymphocyte equivalent
  - No crop offset (JSON centroids already in ROI-local coords)
  - No crop-region filtering (all nuclei valid in small ROI patches)

Note: `v4`/`v15` are kept as internal method keys and CSV column suffixes (not
renamed to stage1/stage2) — same tradeoff documented in compute_metrics.py.
Only the *input* JSON key names (read from classify_tcga.py's output) use the
stage1/stage2 names, since that's what
classify_tcga.py actually writes.

Outputs (into OUTPUT_DIR):
  - cell_df.csv             Per-cell classification results
  - roi_summary.csv         Per ROI aggregate metrics
  - roi_region_summary.csv  Per (ROI, tissue_region) metrics
  - tissue_aggregate.csv    Per tissue region aggregate metrics
"""

import numpy as np
import os
import json
import re
import xml.etree.ElementTree as ET
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr
from skimage.measure import regionprops
from PIL import Image
from collections import defaultdict


# ============================================================
# CONSTANTS
# ============================================================

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Set LOKI_OT_DATA_ROOT to wherever you've placed the TIGER WSIROIS data (see README "Dataset").
DATA_ROOT   = os.environ.get('LOKI_OT_DATA_ROOT', '/DataMount/xl260/wsirois')
BASE_DIR    = os.path.join(DATA_ROOT, "roi-level-annotations", "tissue-cells")
MASK_DIR    = os.path.join(BASE_DIR, "masks")
MAT_DIR     = os.path.join(BASE_DIR, "selected_tcga_stardist_mat")
CELLVIT_DIR = os.path.join(BASE_DIR, "selected_images_output/loki_ot_classifier_tcga")
XML_DIR     = os.path.join(DATA_ROOT, "wsi-level-annotations", "annotations-tissue-cells-xmls")

OUTPUT_DIR  = os.path.join(REPO_ROOT, "outputs", "tcga_classification")

LYMPH_IDX       = 2  # Lizard class index for lymphocyte
PANOPTILS_TIL_IDX = 3  # PanopTILs class index for TILs
LYMPH_GROUP     = "lymphocytes and plasma cells"

# Classifier names for iteration (order: lizard, panoptils, v4, v15).
# Kept as v4/v15 — see module docstring.
CLF_NAMES = ('lizard', 'panoptils', 'v4', 'v15')

TISSUE_NAMES = {
    0: "Background",
    1: "Invasive tumor",
    2: "Tumor-assoc stroma",
    3: "In-situ tumor",
    4: "Healthy glands",
    5: "Necrosis",
    6: "Inflamed stroma",
    7: "Rest",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_tissue_category(x, y, mask):
    """Get tissue category (1-7) at position (x,y) in mask."""
    x, y = int(x), int(y)
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return mask[y, x]
    return 0


def get_nucleus_id(x, y, nuclei_inst_map):
    """Get nucleus instance ID at position (x,y) in nuclei instance map."""
    x, y = int(x), int(y)
    if 0 <= y < nuclei_inst_map.shape[0] and 0 <= x < nuclei_inst_map.shape[1]:
        return nuclei_inst_map[y, x]
    return 0


def classify(gt, pred):
    """Binary classification label from GT and prediction booleans."""
    if gt and pred:
        return 'TP'
    elif not gt and pred:
        return 'FP'
    elif gt and not pred:
        return 'FN'
    else:
        return 'TN'


def compute_metrics(tp, fp, fn, tn):
    """Compute precision, recall, F1, specificity, accuracy from confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'accuracy': accuracy,
    }


# ============================================================
# XML / ROI PARSING
# ============================================================

def parse_roi_name(filename):
    """Parse prefix and bbox from ROI filename."""
    name = filename.replace('.png', '').replace('.mat', '').replace('.json', '')
    m = re.match(r'^(.+?)_\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]$', name)
    if not m:
        return None, None
    prefix = m.group(1)
    bbox = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return prefix, bbox


def parse_cell_annotations(xml_path):
    """Parse ASAP XML -> list of (group, centroid_x, centroid_y) in WSI coords."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cells = []
    for annotation in root.findall('.//Annotation'):
        group = annotation.get('PartOfGroup', '')
        coords = annotation.findall('.//Coordinate')
        if len(coords) == 0:
            continue
        xs = [float(c.get('X')) for c in coords]
        ys = [float(c.get('Y')) for c in coords]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cells.append((group, cx, cy))
    return cells


def get_lymphocyte_centroids(xml_path):
    """Extract lymphocyte centroids from ASAP XML."""
    cells = parse_cell_annotations(xml_path)
    return [(cx, cy) for group, cx, cy in cells if group == LYMPH_GROUP]


def build_xml_lookup():
    """Build mapping from WSI prefix -> XML path."""
    lookup = {}
    for f in os.listdir(XML_DIR):
        if f.endswith('.xml'):
            prefix = f.replace('.xml', '')
            lookup[prefix] = os.path.join(XML_DIR, f)
    return lookup


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Build XML lookup
    # ----------------------------------------------------------
    print("Building XML lookup...")
    xml_lookup = build_xml_lookup()
    print(f"  {len(xml_lookup)} XML files indexed")

    # ----------------------------------------------------------
    # 2. Get ROI list from CellViT TCGA JSON directory
    # ----------------------------------------------------------
    roi_names = sorted([
        f.replace('.json', '')
        for f in os.listdir(CELLVIT_DIR)
        if f.endswith('.json')
    ])
    print(f"Found {len(roi_names)} TCGA ROIs to process")

    # ----------------------------------------------------------
    # 3. Group ROIs by patient (parse XML once per patient)
    # ----------------------------------------------------------
    patient_rois = defaultdict(list)
    n_no_prefix = 0
    n_no_xml = 0
    for roi_name in roi_names:
        prefix, bbox = parse_roi_name(roi_name)
        if prefix is None:
            n_no_prefix += 1
            continue
        if prefix not in xml_lookup:
            n_no_xml += 1
            continue
        patient_rois[prefix].append((roi_name, bbox))

    total_rois = sum(len(v) for v in patient_rois.values())
    print(f"  {total_rois} ROIs across {len(patient_rois)} patients with XML")
    if n_no_prefix > 0:
        print(f"  WARNING: {n_no_prefix} ROIs failed prefix parsing")
    if n_no_xml > 0:
        print(f"  WARNING: {n_no_xml} ROIs missing XML (skipped)")

    # ----------------------------------------------------------
    # 4. Process patient by patient, ROI by ROI
    # ----------------------------------------------------------
    all_cell_results = []
    all_roi_summaries = []
    unmapped_totals = {
        'gt_mapped': 0, 'gt_unmapped': 0,
        'cellvit_mapped': 0, 'cellvit_unmapped': 0,
    }

    patients = sorted(patient_rois.keys())
    print(f"\n{'='*90}")
    print(f"Processing {total_rois} ROIs across {len(patients)} patients")
    print(f"{'='*90}")

    for pi, wsi_prefix in enumerate(patients):
        rois = patient_rois[wsi_prefix]

        # Parse XML once per patient
        lymph_centroids_wsi = get_lymphocyte_centroids(xml_lookup[wsi_prefix])

        for roi_name, bbox in rois:
            x1, y1, x2, y2 = bbox
            png_name = roi_name + '.png'

            # --- Load stardist inst_map from .mat ---
            mat_path = os.path.join(MAT_DIR, roi_name + '.mat')
            if not os.path.exists(mat_path):
                print(f"  WARN: .mat not found for {roi_name}, skipping")
                continue

            mat = sio.loadmat(mat_path)
            inst_map = mat['inst_map']

            # Build nuclei info from regionprops
            props = regionprops(inst_map)
            nuclei_info = {}
            for prop in props:
                nuclei_info[prop.label] = {
                    'centroid': prop.centroid,  # (y, x)
                    'area': prop.area,
                    'gt_lymphocyte': False,
                    'lizard_pred_lymphocyte': False,
                    'panoptils_pred_lymphocyte': False,
                    'v4_pred_lymphocyte': False,
                    'v15_pred_lymphocyte': False,
                }

            # --- Load tissue mask ---
            mask_path = os.path.join(MASK_DIR, png_name)
            if os.path.exists(mask_path):
                tissue_mask = np.array(Image.open(mask_path))
            else:
                tissue_mask = np.zeros(inst_map.shape[:2], dtype=np.uint8)

            # --- Map GT lymphocyte centroids to stardist nuclei ---
            gt_mapped = 0
            gt_unmapped = 0
            for cx_wsi, cy_wsi in lymph_centroids_wsi:
                if x1 <= cx_wsi <= x2 and y1 <= cy_wsi <= y2:
                    rx = cx_wsi - x1
                    ry = cy_wsi - y1
                    nid = get_nucleus_id(rx, ry, inst_map)
                    if nid > 0 and nid in nuclei_info:
                        nuclei_info[nid]['gt_lymphocyte'] = True
                        gt_mapped += 1
                    else:
                        gt_unmapped += 1

            unmapped_totals['gt_mapped'] += gt_mapped
            unmapped_totals['gt_unmapped'] += gt_unmapped

            # --- Load CellViT JSON and map predictions ---
            cellvit_path = os.path.join(CELLVIT_DIR, roi_name + '.json')
            if not os.path.exists(cellvit_path):
                print(f"  WARN: JSON not found for {roi_name}, skipping")
                continue

            with open(cellvit_path) as f:
                cellvit_data = json.load(f)

            cellvit_mapped = 0
            cellvit_unmapped = 0
            for cell_id, cell in cellvit_data['cells'].items():
                cy, cx = cell['centroid']  # [cy, cx] in ROI-local coords
                nid = get_nucleus_id(cx, cy, inst_map)
                if nid > 0 and nid in nuclei_info:
                    if cell['lizard_baseline']['type'] == LYMPH_IDX:
                        nuclei_info[nid]['lizard_pred_lymphocyte'] = True
                    if cell['panoptils_baseline']['type'] == PANOPTILS_TIL_IDX:
                        nuclei_info[nid]['panoptils_pred_lymphocyte'] = True
                    if cell['stage1_context']['type'] == LYMPH_IDX:
                        nuclei_info[nid]['v4_pred_lymphocyte'] = True
                    if cell['stage2_loki_ot']['type'] == LYMPH_IDX:
                        nuclei_info[nid]['v15_pred_lymphocyte'] = True
                    cellvit_mapped += 1
                else:
                    cellvit_unmapped += 1

            unmapped_totals['cellvit_mapped'] += cellvit_mapped
            unmapped_totals['cellvit_unmapped'] += cellvit_unmapped

            # --- Classify each nucleus ---
            cell_results = []
            for nid, info in nuclei_info.items():
                cy, cx = info['centroid']
                tissue_cat = get_tissue_category(cx, cy, tissue_mask)
                if tissue_cat == 0:
                    continue

                gt     = info['gt_lymphocyte']
                liz    = info['lizard_pred_lymphocyte']
                panop  = info['panoptils_pred_lymphocyte']
                v4     = info['v4_pred_lymphocyte']
                v15    = info['v15_pred_lymphocyte']

                cell_results.append({
                    'image': png_name,
                    'nucleus_id': nid,
                    'center_x': cx,
                    'center_y': cy,
                    'tissue_region': int(tissue_cat),
                    'gt_lymphocyte': gt,
                    'lizard_pred_lymphocyte': liz,
                    'panoptils_pred_lymphocyte': panop,
                    'v4_pred_lymphocyte': v4,
                    'v15_pred_lymphocyte': v15,
                    'lizard_classification': classify(gt, liz),
                    'panoptils_classification': classify(gt, panop),
                    'v4_classification': classify(gt, v4),
                    'v15_classification': classify(gt, v15),
                    'area': info['area'],
                })

            all_cell_results.extend(cell_results)

            # --- Per-ROI summary ---
            if cell_results:
                roi_df = pd.DataFrame(cell_results)

                summary = {'image': png_name, 'total_nuclei': len(cell_results),
                           'gt_lymphocytes': gt_mapped + gt_unmapped,
                           'gt_mapped': gt_mapped,
                           'cellvit_mapped': cellvit_mapped,
                           'cellvit_unmapped': cellvit_unmapped}

                for clf_name in CLF_NAMES:
                    col = f'{clf_name}_classification'
                    tp = (roi_df[col] == 'TP').sum()
                    fp = (roi_df[col] == 'FP').sum()
                    fn = (roi_df[col] == 'FN').sum()
                    tn = (roi_df[col] == 'TN').sum()
                    m = compute_metrics(tp, fp, fn, tn)
                    summary.update({
                        f'{clf_name}_TP': tp, f'{clf_name}_FP': fp,
                        f'{clf_name}_FN': fn, f'{clf_name}_TN': tn,
                        f'{clf_name}_precision': m['precision'],
                        f'{clf_name}_recall': m['recall'],
                        f'{clf_name}_f1': m['f1'],
                    })

                # F1 deltas vs lizard
                summary['panoptils_vs_lizard_f1_delta'] = summary['panoptils_f1'] - summary['lizard_f1']
                summary['v4_vs_lizard_f1_delta'] = summary['v4_f1'] - summary['lizard_f1']
                summary['v15_vs_lizard_f1_delta'] = summary['v15_f1'] - summary['lizard_f1']

                if summary['v15_vs_lizard_f1_delta'] > 0.01:
                    summary['improvement'] = 'improved'
                elif summary['v15_vs_lizard_f1_delta'] < -0.01:
                    summary['improvement'] = 'worsened'
                else:
                    summary['improvement'] = 'unchanged'

                all_roi_summaries.append(summary)

            print(f"  [{pi+1}/{len(patients)}] {roi_name[:70]:<70s} "
                  f"nuclei={len(cell_results):4d} gt_map={gt_mapped}/{gt_mapped+gt_unmapped} "
                  f"cv_map={cellvit_mapped}/{cellvit_mapped+cellvit_unmapped}")

        if (pi + 1) % 20 == 0:
            print(f"  --- {pi+1}/{len(patients)} patients done ---")

    # ----------------------------------------------------------
    # 5. Build and save output DataFrames
    # ----------------------------------------------------------
    print(f"\n{'='*90}")
    print("Saving results...")
    print(f"{'='*90}\n")

    cell_df = pd.DataFrame(all_cell_results)
    cell_df.to_csv(os.path.join(OUTPUT_DIR, 'cell_df.csv'), index=False)
    print(f"Saved cell_df.csv ({len(cell_df)} rows)")

    roi_summary_df = pd.DataFrame(all_roi_summaries)
    roi_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'roi_summary.csv'), index=False)
    print(f"Saved roi_summary.csv ({len(roi_summary_df)} rows)")

    # Per (ROI, tissue_region) summary
    roi_region_rows = []
    for region in range(1, 8):
        region_cells = cell_df[cell_df['tissue_region'] == region]
        for img_name, img_group in region_cells.groupby('image'):
            row = {'image': img_name, 'tissue_region': region}
            for clf_name in CLF_NAMES:
                col = f'{clf_name}_classification'
                tp = (img_group[col] == 'TP').sum()
                fp = (img_group[col] == 'FP').sum()
                fn = (img_group[col] == 'FN').sum()
                tn = (img_group[col] == 'TN').sum()
                m = compute_metrics(tp, fp, fn, tn)
                row.update({
                    f'{clf_name}_TP': tp, f'{clf_name}_FP': fp,
                    f'{clf_name}_FN': fn, f'{clf_name}_TN': tn,
                    f'{clf_name}_precision': m['precision'],
                    f'{clf_name}_recall': m['recall'],
                    f'{clf_name}_f1': m['f1'],
                })
            # GT count for this (ROI, tissue) = TP + FN (using lizard, same GT for all)
            row['gt_count'] = row['lizard_TP'] + row['lizard_FN']
            row['total_nuclei'] = len(img_group)
            for clf_name in CLF_NAMES:
                row[f'{clf_name}_pred_count'] = row[f'{clf_name}_TP'] + row[f'{clf_name}_FP']
            row['v15_vs_lizard_f1_delta'] = row['v15_f1'] - row['lizard_f1']
            row['panoptils_vs_lizard_f1_delta'] = row['panoptils_f1'] - row['lizard_f1']
            roi_region_rows.append(row)

    roi_region_df = pd.DataFrame(roi_region_rows)
    roi_region_df.to_csv(os.path.join(OUTPUT_DIR, 'roi_region_summary.csv'), index=False)
    print(f"Saved roi_region_summary.csv ({len(roi_region_df)} rows)")

    # Tissue aggregate
    tissue_rows = []
    for region in range(1, 8):
        region_cells = cell_df[cell_df['tissue_region'] == region]
        if len(region_cells) == 0:
            continue

        row = {
            'tissue_region': region,
            'tissue_name': TISSUE_NAMES.get(region, f'Region_{region}'),
            'total_nuclei': len(region_cells),
        }
        for clf_name in CLF_NAMES:
            col = f'{clf_name}_classification'
            tp = (region_cells[col] == 'TP').sum()
            fp = (region_cells[col] == 'FP').sum()
            fn = (region_cells[col] == 'FN').sum()
            tn = (region_cells[col] == 'TN').sum()
            m = compute_metrics(tp, fp, fn, tn)
            row.update({
                f'{clf_name}_TP': tp, f'{clf_name}_FP': fp,
                f'{clf_name}_FN': fn, f'{clf_name}_TN': tn,
                f'{clf_name}_precision': round(m['precision'], 4),
                f'{clf_name}_recall': round(m['recall'], 4),
                f'{clf_name}_f1': round(m['f1'], 4),
            })
        row['v15_vs_lizard_f1_delta'] = round(row['v15_f1'] - row['lizard_f1'], 4)
        row['panoptils_vs_lizard_f1_delta'] = round(row['panoptils_f1'] - row['lizard_f1'], 4)
        tissue_rows.append(row)

    tissue_agg_df = pd.DataFrame(tissue_rows)
    tissue_agg_df.to_csv(os.path.join(OUTPUT_DIR, 'tissue_aggregate.csv'), index=False)
    print(f"Saved tissue_aggregate.csv ({len(tissue_agg_df)} rows)")

    # ----------------------------------------------------------
    # 6. Console summary
    # ----------------------------------------------------------
    print(f"\n{'='*150}")
    print("AGGREGATE METRICS BY TISSUE REGION")
    print(f"{'='*150}")
    print(f"{'Region':<20s} {'Nuclei':>7s} | "
          f"{'Liz_P':>7s} {'Liz_R':>7s} {'Liz_F1':>7s} | "
          f"{'Pan_P':>7s} {'Pan_R':>7s} {'Pan_F1':>7s} | "
          f"{'V4_P':>7s} {'V4_R':>7s} {'V4_F1':>7s} | "
          f"{'V15_P':>7s} {'V15_R':>7s} {'V15_F1':>7s} | "
          f"{'dF1_p':>7s} {'dF1_15':>7s}")
    print("-" * 150)
    for _, row in tissue_agg_df.iterrows():
        print(f"{row['tissue_name']:<20s} {row['total_nuclei']:>7d} | "
              f"{row['lizard_precision']:>7.4f} {row['lizard_recall']:>7.4f} {row['lizard_f1']:>7.4f} | "
              f"{row['panoptils_precision']:>7.4f} {row['panoptils_recall']:>7.4f} {row['panoptils_f1']:>7.4f} | "
              f"{row['v4_precision']:>7.4f} {row['v4_recall']:>7.4f} {row['v4_f1']:>7.4f} | "
              f"{row['v15_precision']:>7.4f} {row['v15_recall']:>7.4f} {row['v15_f1']:>7.4f} | "
              f"{row['panoptils_vs_lizard_f1_delta']:>+7.4f} {row['v15_vs_lizard_f1_delta']:>+7.4f}")

    # Overall aggregate
    print("-" * 150)
    overall = {'total': len(cell_df)}
    for clf_name in CLF_NAMES:
        col = f'{clf_name}_classification'
        tp = cell_df[col].eq('TP').sum()
        fp = cell_df[col].eq('FP').sum()
        fn = cell_df[col].eq('FN').sum()
        tn = cell_df[col].eq('TN').sum()
        overall[clf_name] = compute_metrics(tp, fp, fn, tn)

    print(f"{'TOTAL':<20s} {overall['total']:>7d} | "
          f"{overall['lizard']['precision']:>7.4f} {overall['lizard']['recall']:>7.4f} {overall['lizard']['f1']:>7.4f} | "
          f"{overall['panoptils']['precision']:>7.4f} {overall['panoptils']['recall']:>7.4f} {overall['panoptils']['f1']:>7.4f} | "
          f"{overall['v4']['precision']:>7.4f} {overall['v4']['recall']:>7.4f} {overall['v4']['f1']:>7.4f} | "
          f"{overall['v15']['precision']:>7.4f} {overall['v15']['recall']:>7.4f} {overall['v15']['f1']:>7.4f} | "
          f"{overall['panoptils']['f1'] - overall['lizard']['f1']:>+7.4f} "
          f"{overall['v15']['f1'] - overall['lizard']['f1']:>+7.4f}")

    # ROI improvement tracking
    print(f"\n{'='*90}")
    print("ROI IMPROVEMENT TRACKING (lizard baseline -> v15 OT Distilled)")
    print(f"{'='*90}")
    n_improved = (roi_summary_df['improvement'] == 'improved').sum()
    n_unchanged = (roi_summary_df['improvement'] == 'unchanged').sum()
    n_worsened = (roi_summary_df['improvement'] == 'worsened').sum()
    n_total = len(roi_summary_df)
    print(f"  Improved:  {n_improved}/{n_total} ({100*n_improved/n_total:.1f}%)")
    print(f"  Unchanged: {n_unchanged}/{n_total} ({100*n_unchanged/n_total:.1f}%)")
    print(f"  Worsened:  {n_worsened}/{n_total} ({100*n_worsened/n_total:.1f}%)")

    # Unmapped statistics
    print(f"\n{'='*90}")
    print("MAPPING STATISTICS")
    print(f"{'='*90}")
    gt_total = unmapped_totals['gt_mapped'] + unmapped_totals['gt_unmapped']
    cv_total = unmapped_totals['cellvit_mapped'] + unmapped_totals['cellvit_unmapped']
    print(f"  GT annotations:  {unmapped_totals['gt_mapped']}/{gt_total} mapped "
          f"({unmapped_totals['gt_unmapped']} unmapped)")
    print(f"  CellViT cells:   {unmapped_totals['cellvit_mapped']}/{cv_total} mapped "
          f"({unmapped_totals['cellvit_unmapped']} unmapped)")

    # ----------------------------------------------------------
    # 7. COUNT-LEVEL METRICS: MAE, Pearson r, FP% reduction
    # ----------------------------------------------------------

    # Per-ROI predicted lymphocyte count = TP + FP; GT count = gt_mapped
    for clf_name in CLF_NAMES:
        roi_summary_df[f'{clf_name}_pred_count'] = (
            roi_summary_df[f'{clf_name}_TP'] + roi_summary_df[f'{clf_name}_FP']
        )

    # --- MAE (count-level, per ROI) — global ---
    print(f"\n{'='*90}")
    print("COUNT-LEVEL MAE (mean |pred_count - gt_mapped| per ROI)")
    print(f"{'='*90}")
    for clf_name in CLF_NAMES:
        mae = (roi_summary_df[f'{clf_name}_pred_count'] - roi_summary_df['gt_mapped']).abs().mean()
        print(f"  {clf_name:>10s}  MAE = {mae:.2f}")

    # --- Pearson r (count-level, per ROI) — global ---
    print(f"\n{'='*90}")
    print("COUNT-LEVEL PEARSON r (pred_count vs gt_mapped per ROI)")
    print(f"{'='*90}")
    gt_counts = roi_summary_df['gt_mapped'].values.astype(float)
    for clf_name in CLF_NAMES:
        pred_counts = roi_summary_df[f'{clf_name}_pred_count'].values.astype(float)
        r, p = pearsonr(gt_counts, pred_counts)
        print(f"  {clf_name:>10s}  r = {r:.4f}  (p = {p:.2e})")

    # --- MAE & r BY TISSUE TYPE (per ROI-tissue block) ---
    print(f"\n{'='*140}")
    print("COUNT-LEVEL MAE & PEARSON r BY TISSUE TYPE")
    print(f"{'='*140}")
    print(f"  {'Region':<20s} {'n_blocks':>8s} | "
          f"{'Liz_MAE':>8s} {'Pan_MAE':>8s} {'V4_MAE':>8s} {'V15_MAE':>8s} | "
          f"{'Liz_r':>8s} {'Pan_r':>8s} {'V4_r':>8s} {'V15_r':>8s}")
    print("  " + "-" * 120)
    for region in range(1, 8):
        rr = roi_region_df[roi_region_df['tissue_region'] == region]
        if len(rr) == 0:
            continue
        tname = TISSUE_NAMES.get(region, f'Region_{region}')
        gt_t = rr['gt_count'].values.astype(float)
        mae_vals = {}
        r_vals = {}
        for clf_name in CLF_NAMES:
            pred_t = rr[f'{clf_name}_pred_count'].values.astype(float)
            mae_vals[clf_name] = np.abs(pred_t - gt_t).mean()
            if len(gt_t) > 2 and gt_t.std() > 0:
                r_vals[clf_name], _ = pearsonr(gt_t, pred_t)
            else:
                r_vals[clf_name] = float('nan')
        print(f"  {tname:<20s} {len(rr):>8d} | "
              f"{mae_vals['lizard']:>8.2f} {mae_vals['panoptils']:>8.2f} "
              f"{mae_vals['v4']:>8.2f} {mae_vals['v15']:>8.2f} | "
              f"{r_vals['lizard']:>8.4f} {r_vals['panoptils']:>8.4f} "
              f"{r_vals['v4']:>8.4f} {r_vals['v15']:>8.4f}")

    # --- FP% reduction (vs lizard baseline) ---
    print(f"\n{'='*100}")
    print("FP% REDUCTION vs LIZARD BASELINE (per tissue)")
    print(f"{'='*100}")
    print(f"  {'Region':<20s} {'Liz_FP':>8s} {'Pan_FP':>8s} {'V4_FP':>8s} {'V15_FP':>8s} | "
          f"{'Pan_red%':>9s} {'V4_red%':>8s} {'V15_red%':>9s}")
    print("  " + "-" * 85)
    for _, row in tissue_agg_df.iterrows():
        liz_fp   = row['lizard_FP']
        panop_fp = row['panoptils_FP']
        v4_fp    = row['v4_FP']
        v15_fp   = row['v15_FP']
        panop_red = (liz_fp - panop_fp) / liz_fp * 100 if liz_fp > 0 else 0
        v4_red    = (liz_fp - v4_fp) / liz_fp * 100 if liz_fp > 0 else 0
        v15_red   = (liz_fp - v15_fp) / liz_fp * 100 if liz_fp > 0 else 0
        print(f"  {row['tissue_name']:<20s} {liz_fp:>8d} {panop_fp:>8d} {v4_fp:>8d} {v15_fp:>8d} | "
              f"{panop_red:>+8.1f}% {v4_red:>+7.1f}% {v15_red:>+8.1f}%")

    # Global FP reduction
    liz_fp_total   = cell_df['lizard_classification'].eq('FP').sum()
    panop_fp_total = cell_df['panoptils_classification'].eq('FP').sum()
    v4_fp_total    = cell_df['v4_classification'].eq('FP').sum()
    v15_fp_total   = cell_df['v15_classification'].eq('FP').sum()
    panop_red_total = (liz_fp_total - panop_fp_total) / liz_fp_total * 100 if liz_fp_total > 0 else 0
    v4_red_total    = (liz_fp_total - v4_fp_total) / liz_fp_total * 100 if liz_fp_total > 0 else 0
    v15_red_total   = (liz_fp_total - v15_fp_total) / liz_fp_total * 100 if liz_fp_total > 0 else 0
    print("  " + "-" * 85)
    print(f"  {'TOTAL':<20s} {liz_fp_total:>8d} {panop_fp_total:>8d} {v4_fp_total:>8d} {v15_fp_total:>8d} | "
          f"{panop_red_total:>+8.1f}% {v4_red_total:>+7.1f}% {v15_red_total:>+8.1f}%")

    # --- TP% loss (vs lizard baseline) — to see recall cost ---
    print(f"\n{'='*100}")
    print("TP% LOSS vs LIZARD BASELINE (per tissue)")
    print(f"{'='*100}")
    print(f"  {'Region':<20s} {'Liz_TP':>8s} {'Pan_TP':>8s} {'V4_TP':>8s} {'V15_TP':>8s} | "
          f"{'Pan_loss%':>10s} {'V4_loss%':>9s} {'V15_loss%':>10s}")
    print("  " + "-" * 90)
    for _, row in tissue_agg_df.iterrows():
        liz_tp   = row['lizard_TP']
        panop_tp = row['panoptils_TP']
        v4_tp    = row['v4_TP']
        v15_tp   = row['v15_TP']
        panop_loss = (liz_tp - panop_tp) / liz_tp * 100 if liz_tp > 0 else 0
        v4_loss    = (liz_tp - v4_tp) / liz_tp * 100 if liz_tp > 0 else 0
        v15_loss   = (liz_tp - v15_tp) / liz_tp * 100 if liz_tp > 0 else 0
        print(f"  {row['tissue_name']:<20s} {liz_tp:>8d} {panop_tp:>8d} {v4_tp:>8d} {v15_tp:>8d} | "
              f"{panop_loss:>+9.1f}% {v4_loss:>+8.1f}% {v15_loss:>+9.1f}%")

    liz_tp_total   = cell_df['lizard_classification'].eq('TP').sum()
    panop_tp_total = cell_df['panoptils_classification'].eq('TP').sum()
    v4_tp_total    = cell_df['v4_classification'].eq('TP').sum()
    v15_tp_total   = cell_df['v15_classification'].eq('TP').sum()
    panop_loss_total = (liz_tp_total - panop_tp_total) / liz_tp_total * 100 if liz_tp_total > 0 else 0
    v4_loss_total    = (liz_tp_total - v4_tp_total) / liz_tp_total * 100 if liz_tp_total > 0 else 0
    v15_loss_total   = (liz_tp_total - v15_tp_total) / liz_tp_total * 100 if liz_tp_total > 0 else 0
    print("  " + "-" * 90)
    print(f"  {'TOTAL':<20s} {liz_tp_total:>8d} {panop_tp_total:>8d} {v4_tp_total:>8d} {v15_tp_total:>8d} | "
          f"{panop_loss_total:>+9.1f}% {v4_loss_total:>+8.1f}% {v15_loss_total:>+9.1f}%")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
