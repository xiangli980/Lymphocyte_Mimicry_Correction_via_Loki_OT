"""
TCGA holdout evaluation: Lizard / PanopTILs / Context-soft (v4) / Loki-OT (v15).

  - 4 methods (Lizard/PanopTILs baselines + Context-soft/Loki-OT from this repo)
  - Bootstrap 95% CI on MAE / F1 / Precision / Recall, at both ROI and patient level
  - Paired Wilcoxon (Loki-OT-anchored, 3 comparisons each) at ROI and patient level
  - Per-tissue MAE + bootstrap CI at ROI level (and per-patient where >= 30 patients)
  - PanopTILs uses class index 3 (TILs); Lizard / v4 / v15 use index 2.

GT canonical: TP+FN (matches F1/P/R basis; per-tissue gt_mapped is unavailable
in the existing csv layout — see docs/eval_tables_summary.md "Per-tissue analysis" note).

Note: `v4`/`v15` are kept as internal method keys (not renamed to stage1/stage2)
because they map 1:1 to the `v4_*`/`v15_*` column names already baked into the
aggregate CSVs this script reads (see TCGA_DIR below).

Input: expects the 4 aggregate CSVs (tissue_aggregate.csv, roi_summary.csv,
roi_region_summary.csv, cell_df.csv) in TCGA_DIR below, plus per-cell JSON
predictions in JSON_DIR. Both are produced by eval/aggregate_tcga_predictions.py
(run that first).

Outputs: outputs/eval_tables/ + docs/eval_tables_summary.md.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


# ── Constants ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TCGA_DIR = os.path.join(REPO_ROOT, "outputs", "tcga_classification")
# Set LOKI_OT_DATA_ROOT to wherever you've placed the TIGER WSIROIS data (see README "Dataset").
DATA_ROOT = os.environ.get('LOKI_OT_DATA_ROOT', '/DataMount/xl260/wsirois')
BASE_DIR = os.path.join(DATA_ROOT, "roi-level-annotations", "tissue-cells") + "/"
JSON_DIR = os.path.join(BASE_DIR, "selected_images_output", "loki_ot_classifier_tcga")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "eval_tables")

METHODS = ["lizard", "panoptils", "v4", "v15"]
LOKI = "v15"  # anchor for pairwise Wilcoxon
METHOD_LABELS = {
    "lizard": "Lizard",
    "panoptils": "PanopTILs",
    "v4": "Context-soft (v4)",
    "v15": "Loki-OT (v15)",
}

# Method-specific lymphocyte class index in 6-class softmax.
# Lizard schema: class 2 = lymphocyte; PanopTILs schema: class 3 = TILs.
LYMPH_IDX = {"lizard": 2, "panoptils": 3, "v4": 2, "v15": 2}
# JSON keys as written by src/classify_tcga.py (stage1_context/stage2_loki_ot,
# renamed from v4_context/v15_distilled during that script's migration).
JSON_KEY = {
    "lizard": "lizard_baseline",
    "panoptils": "panoptils_baseline",
    "v4": "stage1_context",
    "v15": "stage2_loki_ot",
}

TISSUE_NAMES = {
    1: "T1 Invasive tumor",
    2: "T2 Tumor-assoc stroma",
    3: "T3 In-situ tumor",
    4: "T4 Healthy glands",
    5: "T5 Necrosis",
    6: "T6 Inflamed stroma",
    7: "T7 Rest",
}

N_BOOT = 1000
RNG_SEED = 42
PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")
PATIENT_TISSUE_MIN = 30  # min patients with that tissue for per-patient per-tissue analysis

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def f1_score(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def prec_score(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def rec_score(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def pooled_metric(tp_arr, fp_arr, fn_arr, metric):
    """Vectorized pooled metric over bootstrap resample dim (axis 0)."""
    tp = tp_arr.sum(axis=-1)
    fp = fp_arr.sum(axis=-1)
    fn = fn_arr.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "precision":
            return np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        if metric == "recall":
            return np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        if metric == "f1":
            denom = 2 * tp + fp + fn
            return np.where(denom > 0, 2 * tp / denom, 0.0)
    raise ValueError(metric)


def abs_errors(df, method):
    """|pred - gt| per row. GT = lizard_TP + lizard_FN (canonical TP+FN)."""
    gt = (df["lizard_TP"] + df["lizard_FN"]).to_numpy()
    pred = (df[f"{method}_TP"] + df[f"{method}_FP"]).to_numpy()
    return np.abs(pred - gt).astype(float)


def extract_patient_id(image_name):
    m = PATIENT_RE.match(image_name)
    if m is None:
        raise ValueError(f"No TCGA patient barcode in {image_name!r}")
    return m.group(1)


def aggregate_to_patient(df_roi):
    """Sum TP/FP/FN/TN columns per patient. Returns 124-row df."""
    df = df_roi.copy()
    df["patient_id"] = df["image"].apply(extract_patient_id)
    sum_cols = [c for c in df.columns
                if any(c.endswith(s) for s in ("_TP", "_FP", "_FN", "_TN"))]
    extra_cols = [c for c in ["total_nuclei", "gt_lymphocytes", "gt_mapped"] if c in df.columns]
    return df.groupby("patient_id", as_index=False)[sum_cols + extra_cols].sum()


def bootstrap_ci_mae(errors, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile bootstrap CI on mean of `errors` array."""
    if len(errors) == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    n = len(errors)
    boots = rng.choice(errors, size=(n_boot, n), replace=True).mean(axis=1)
    return {
        "point": float(errors.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "n": n,
    }


def bootstrap_ci_pooled(df, method, metric, n_boot=N_BOOT, seed=RNG_SEED):
    """Bootstrap CI on pooled F1/Precision/Recall (resample rows, sum TP/FP/FN, recompute)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    tp = df[f"{method}_TP"].to_numpy()
    fp = df[f"{method}_FP"].to_numpy()
    fn = df[f"{method}_FN"].to_numpy()
    idx = rng.choice(n, size=(n_boot, n), replace=True)
    boots = pooled_metric(tp[idx], fp[idx], fn[idx], metric)
    point_full = pooled_metric(tp[None, :], fp[None, :], fn[None, :], metric)[0]
    return {
        "point": float(point_full),
        "ci_low": float(np.nanquantile(boots, 0.025)),
        "ci_high": float(np.nanquantile(boots, 0.975)),
        "n": n,
    }


def bootstrap_ci_apauc(df_cells_sub, score_col, metric_func, n_boot=N_BOOT, seed=RNG_SEED):
    """ROI-level bootstrap of a cell-score-based metric (AP or AUC).

    Bootstrap unit is the ROI (image), matching the bootstrap-unit convention used
    elsewhere in this script. For each iteration: resample ROIs with replacement,
    take all cells in resampled ROIs, compute the metric on the resampled cell set.

    df_cells_sub: per-cell df with image, gt_lymphocyte (bool), and {score_col} cols.
    metric_func: callable(gt_int_array, score_array) -> float
    """
    if len(df_cells_sub) == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_cells": 0, "n_roi": 0}
    gt_full = df_cells_sub["gt_lymphocyte"].astype(int).to_numpy()
    scores_full = df_cells_sub[score_col].to_numpy()
    n_pos_full = int(gt_full.sum())
    if n_pos_full == 0 or n_pos_full == len(gt_full):
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_cells": len(df_cells_sub), "n_roi": df_cells_sub["image"].nunique()}

    point = float(metric_func(gt_full, scores_full))

    rois = df_cells_sub["image"].to_numpy()
    unique_rois = np.unique(rois)
    n_rois = len(unique_rois)
    roi_to_indices = {r: np.where(rois == r)[0] for r in unique_rois}

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.integers(0, n_rois, size=n_rois)
        cell_indices = np.concatenate([roi_to_indices[unique_rois[i]] for i in sampled])
        gt_b = gt_full[cell_indices]
        if gt_b.sum() == 0 or gt_b.sum() == len(gt_b):
            boots[b] = float("nan")
        else:
            boots[b] = float(metric_func(gt_b, scores_full[cell_indices]))

    return {
        "point": point,
        "ci_low": float(np.nanquantile(boots, 0.025)),
        "ci_high": float(np.nanquantile(boots, 0.975)),
        "n_cells": len(df_cells_sub),
        "n_roi": n_rois,
    }


def paired_wilcoxon(err_a, err_b, alternative, n_boot=N_BOOT, seed=RNG_SEED):
    """Paired Wilcoxon on (err_a - err_b) + bootstrap CI on median diff."""
    diff = err_a - err_b
    if len(diff) == 0 or np.all(diff == 0):
        return {
            "stat": float("nan"), "p_value": float("nan"),
            "median_diff": 0.0, "median_ci_low": 0.0, "median_ci_high": 0.0,
            "n": len(diff), "alternative": alternative,
        }
    try:
        res = wilcoxon(err_a, err_b, alternative=alternative, zero_method="wilcox")
        stat, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        stat, p = float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(diff, size=(n_boot, len(diff)), replace=True), axis=1)
    return {
        "stat": stat,
        "p_value": p,
        "median_diff": float(np.median(diff)),
        "median_ci_low": float(np.quantile(boots, 0.025)),
        "median_ci_high": float(np.quantile(boots, 0.975)),
        "n": len(diff),
        "alternative": alternative,
    }


# ── Load data ────────────────────────────────────────────────────────────────
print(f"Loading data from {TCGA_DIR}...")
df_tissue = pd.read_csv(os.path.join(TCGA_DIR, "tissue_aggregate.csv"))
df_roi = pd.read_csv(os.path.join(TCGA_DIR, "roi_summary.csv"))
df_roi_tissue = pd.read_csv(os.path.join(TCGA_DIR, "roi_region_summary.csv"))

n_rois = len(df_roi)
n_nuclei = int(df_tissue["total_nuclei"].sum())

df_patient = aggregate_to_patient(df_roi)
n_patients = len(df_patient)
print(f"  ROIs: {n_rois}; nuclei: {n_nuclei:,}; patients: {n_patients}")
assert n_patients == 124, f"Expected 124 patients, got {n_patients}"

df_patient.to_csv(os.path.join(OUT_DIR, "patient_aggregate.csv"), index=False)


# ── §1. Raw counts ───────────────────────────────────────────────────────────
rows_counts = []
for _, r in df_tissue.iterrows():
    tissue = int(r["tissue_region"])
    gt = int(r["lizard_TP"] + r["lizard_FN"])
    row = dict(tissue=tissue, tissue_name=TISSUE_NAMES[tissue],
               total_nuclei=int(r["total_nuclei"]), gt_lymph=gt)
    for m in METHODS:
        row[f"{m}_pred"] = int(r[f"{m}_TP"] + r[f"{m}_FP"])
    rows_counts.append(row)
row_g = dict(tissue=0, tissue_name="Global", total_nuclei=n_nuclei,
             gt_lymph=int((df_tissue["lizard_TP"] + df_tissue["lizard_FN"]).sum()))
for m in METHODS:
    row_g[f"{m}_pred"] = int((df_tissue[f"{m}_TP"] + df_tissue[f"{m}_FP"]).sum())
rows_counts.insert(0, row_g)
df_counts = pd.DataFrame(rows_counts)
df_counts.to_csv(os.path.join(OUT_DIR, "raw_counts.csv"), index=False)


# ── §2. P/R/F1 per tissue (point estimates) ──────────────────────────────────
rows_prf = []
for _, r in df_tissue.iterrows():
    tissue = int(r["tissue_region"])
    row = dict(tissue=tissue, tissue_name=TISSUE_NAMES[tissue])
    for m in METHODS:
        tp, fp, fn = r[f"{m}_TP"], r[f"{m}_FP"], r[f"{m}_FN"]
        row[f"{m}_prec"] = prec_score(tp, fp)
        row[f"{m}_rec"] = rec_score(tp, fn)
        row[f"{m}_f1"] = f1_score(tp, fp, fn)
    rows_prf.append(row)
row_g = dict(tissue=0, tissue_name="Global")
for m in METHODS:
    tp = df_tissue[f"{m}_TP"].sum(); fp = df_tissue[f"{m}_FP"].sum(); fn = df_tissue[f"{m}_FN"].sum()
    row_g[f"{m}_prec"] = prec_score(tp, fp)
    row_g[f"{m}_rec"] = rec_score(tp, fn)
    row_g[f"{m}_f1"] = f1_score(tp, fp, fn)
rows_prf.insert(0, row_g)
row_macro = dict(tissue=-1, tissue_name="Macro F1")
tissue_rows = [rr for rr in rows_prf if rr["tissue"] > 0]
for m in METHODS:
    row_macro[f"{m}_prec"] = float(np.mean([rr[f"{m}_prec"] for rr in tissue_rows]))
    row_macro[f"{m}_rec"] = float(np.mean([rr[f"{m}_rec"] for rr in tissue_rows]))
    row_macro[f"{m}_f1"] = float(np.mean([rr[f"{m}_f1"] for rr in tissue_rows]))
rows_prf.insert(1, row_macro)
df_prf = pd.DataFrame(rows_prf)
df_prf.to_csv(os.path.join(OUT_DIR, "prf_by_tissue.csv"), index=False)


# ── §3. FP elimination / TP loss / Selectivity ───────────────────────────────
fp_rows = []
global_row = df_tissue.sum(numeric_only=True)
global_row["tissue_region"] = 0
non_baseline = [m for m in METHODS if m != "lizard"]
for source in [(0, global_row)] + list(df_tissue.iterrows()):
    idx, r = source
    tissue = int(r["tissue_region"])
    name = "Global" if tissue == 0 else TISSUE_NAMES[tissue]
    bl_fp = r["lizard_FP"]; bl_tp = r["lizard_TP"]
    for m in non_baseline:
        ver_fp = r[f"{m}_FP"]; ver_tp = r[f"{m}_TP"]
        fp_rem = (bl_fp - ver_fp) / bl_fp * 100 if bl_fp > 0 else 0.0
        tp_lost = (bl_tp - ver_tp) / bl_tp * 100 if bl_tp > 0 else 0.0
        if tp_lost > 0:
            sel = fp_rem / tp_lost
        elif fp_rem > 0:
            sel = float("inf")
        else:
            sel = float("nan")
        fp_rows.append(dict(tissue=tissue, tissue_name=name, version=m,
                            fp_removed_pct=fp_rem, tp_lost_pct=tp_lost, selectivity=sel))
df_fp = pd.DataFrame(fp_rows)
df_fp.to_csv(os.path.join(OUT_DIR, "fp_elimination.csv"), index=False)


# ── §4. TIL density ──────────────────────────────────────────────────────────
rows_dens = []
for _, r in df_tissue.iterrows():
    tissue = int(r["tissue_region"])
    n = r["total_nuclei"]
    gt_dens = (r["lizard_TP"] + r["lizard_FN"]) / n * 100 if n > 0 else 0.0
    row = dict(tissue=tissue, tissue_name=TISSUE_NAMES[tissue], gt_density=gt_dens)
    for m in METHODS:
        row[f"{m}_density"] = (r[f"{m}_TP"] + r[f"{m}_FP"]) / n * 100 if n > 0 else 0.0
    rows_dens.append(row)
df_dens = pd.DataFrame(rows_dens)
df_dens.to_csv(os.path.join(OUT_DIR, "til_density.csv"), index=False)


# ── §5. Per-ROI MAE + 95% CI ─────────────────────────────────────────────────
print("Bootstrapping per-ROI MAE...")
rows_mae_roi = []
for m in METHODS:
    errs = abs_errors(df_roi, m)
    ci = bootstrap_ci_mae(errs)
    rows_mae_roi.append(dict(method=m, label=METHOD_LABELS[m], **ci))
df_mae_roi = pd.DataFrame(rows_mae_roi)
df_mae_roi.to_csv(os.path.join(OUT_DIR, "mae_global_roi.csv"), index=False)


# ── §6. Per-ROI F1 / Precision / Recall + 95% CI ─────────────────────────────
print("Bootstrapping per-ROI F1/Precision/Recall...")
rows_metrics_roi = []
for m in METHODS:
    for metric in ["f1", "precision", "recall"]:
        ci = bootstrap_ci_pooled(df_roi, m, metric)
        rows_metrics_roi.append(dict(method=m, label=METHOD_LABELS[m], metric=metric, **ci))
df_metrics_roi = pd.DataFrame(rows_metrics_roi)
df_metrics_roi.to_csv(os.path.join(OUT_DIR, "metrics_global_roi.csv"), index=False)


# ── §7. Per-tissue MAE + 95% CI (per-ROI) ────────────────────────────────────
print("Bootstrapping per-tissue MAE (per-ROI)...")
rows_mae_t = []
rows_pearson_t = []
for tissue in sorted(df_roi_tissue["tissue_region"].unique()):
    if tissue == 0:
        continue
    sub = df_roi_tissue[df_roi_tissue["tissue_region"] == tissue]
    name = TISSUE_NAMES.get(tissue, f"T{tissue}")
    row_pearson = dict(tissue=tissue, tissue_name=name)
    for m in METHODS:
        errs = abs_errors(sub, m)
        ci = bootstrap_ci_mae(errs, seed=RNG_SEED + tissue)
        rows_mae_t.append(dict(tissue=tissue, tissue_name=name, method=m,
                               label=METHOD_LABELS[m], **ci))
        gt = sub["lizard_TP"] + sub["lizard_FN"]
        pred = sub[f"{m}_TP"] + sub[f"{m}_FP"]
        if len(gt) > 2 and gt.std() > 0 and pred.std() > 0:
            row_pearson[f"{m}_r"] = float(pearsonr(pred, gt)[0])
        else:
            row_pearson[f"{m}_r"] = float("nan")
    rows_pearson_t.append(row_pearson)
df_mae_t = pd.DataFrame(rows_mae_t)
df_mae_t.to_csv(os.path.join(OUT_DIR, "mae_by_tissue_roi.csv"), index=False)


# ── §8. Per-tissue Pearson r ─────────────────────────────────────────────────
df_pearson_t = pd.DataFrame(rows_pearson_t)
df_pearson_t.to_csv(os.path.join(OUT_DIR, "pearson_by_tissue.csv"), index=False)


# ── §8b. Per-tissue F1/Precision/Recall + 95% CI (per-ROI bootstrap) ─────────
print("Bootstrapping per-tissue F1/Precision/Recall (per-ROI)...")
rows_prf_t_ci = []
for tissue in sorted(df_roi_tissue["tissue_region"].unique()):
    if tissue == 0:
        continue
    sub = df_roi_tissue[df_roi_tissue["tissue_region"] == tissue]
    name = TISSUE_NAMES.get(tissue, f"T{tissue}")
    for m in METHODS:
        for mi, metric in enumerate(["f1", "precision", "recall"]):
            ci = bootstrap_ci_pooled(sub, m, metric,
                                      seed=RNG_SEED + tissue * 10 + mi)
            rows_prf_t_ci.append(dict(tissue=tissue, tissue_name=name, method=m,
                                       label=METHOD_LABELS[m], metric=metric, **ci))
df_prf_t_ci = pd.DataFrame(rows_prf_t_ci)
df_prf_t_ci.to_csv(os.path.join(OUT_DIR, "prf_by_tissue_ci.csv"), index=False)


# ── §10. Per-patient MAE + 95% CI ────────────────────────────────────────────
print("Bootstrapping per-patient MAE...")
rows_mae_pat = []
for m in METHODS:
    errs = abs_errors(df_patient, m)
    ci = bootstrap_ci_mae(errs)
    rows_mae_pat.append(dict(method=m, label=METHOD_LABELS[m], **ci))
df_mae_pat = pd.DataFrame(rows_mae_pat)
df_mae_pat.to_csv(os.path.join(OUT_DIR, "mae_global_patient.csv"), index=False)


# ── §11. Per-patient F1/P/R + 95% CI ─────────────────────────────────────────
print("Bootstrapping per-patient F1/Precision/Recall...")
rows_metrics_pat = []
for m in METHODS:
    for metric in ["f1", "precision", "recall"]:
        ci = bootstrap_ci_pooled(df_patient, m, metric)
        rows_metrics_pat.append(dict(method=m, label=METHOD_LABELS[m], metric=metric, **ci))
df_metrics_pat = pd.DataFrame(rows_metrics_pat)
df_metrics_pat.to_csv(os.path.join(OUT_DIR, "metrics_global_patient.csv"), index=False)


# ── §11b. Per-tissue MAE + 95% CI (per-patient) ──────────────────────────────
print("Bootstrapping per-tissue MAE (per-patient)...")
# Aggregate roi_region_summary to patient × tissue
df_rrt = df_roi_tissue.copy()
df_rrt["patient_id"] = df_rrt["image"].apply(extract_patient_id)
sum_cols_rrt = [c for c in df_rrt.columns
                if any(c.endswith(s) for s in ("_TP", "_FP", "_FN", "_TN"))]
df_pat_tissue = (df_rrt.groupby(["patient_id", "tissue_region"], as_index=False)
                 [sum_cols_rrt].sum())

rows_mae_t_pat = []
for tissue in sorted(df_pat_tissue["tissue_region"].unique()):
    if tissue == 0:
        continue
    sub = df_pat_tissue[df_pat_tissue["tissue_region"] == tissue]
    # Filter patients who have non-zero cells in this tissue (TP+FP+FN+TN > 0 for any method)
    has_cells = ((sub["lizard_TP"] + sub["lizard_FP"] + sub["lizard_FN"] + sub["lizard_TN"]) > 0)
    sub = sub[has_cells]
    n_pat = len(sub)
    name = TISSUE_NAMES.get(tissue, f"T{tissue}")
    if n_pat < PATIENT_TISSUE_MIN:
        for m in METHODS:
            rows_mae_t_pat.append(dict(tissue=tissue, tissue_name=name, method=m,
                                        label=METHOD_LABELS[m],
                                        point=float("nan"), ci_low=float("nan"),
                                        ci_high=float("nan"), n=n_pat,
                                        note=f"skipped (only {n_pat} patients < {PATIENT_TISSUE_MIN})"))
        continue
    for m in METHODS:
        errs = abs_errors(sub, m)
        ci = bootstrap_ci_mae(errs, seed=RNG_SEED + 100 + tissue)
        rows_mae_t_pat.append(dict(tissue=tissue, tissue_name=name, method=m,
                                    label=METHOD_LABELS[m], note="", **ci))
df_mae_t_pat = pd.DataFrame(rows_mae_t_pat)
df_mae_t_pat.to_csv(os.path.join(OUT_DIR, "mae_by_tissue_patient.csv"), index=False)


# ── §12-14. Per-cell probability extraction + AP/AUC ─────────────────────────
print("Reading v16 JSONs for AP/AUC...")
df_cells = pd.read_csv(os.path.join(TCGA_DIR, "cell_df.csv"))
prob_map = {}
json_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith(".json"))
for ji, jf in enumerate(json_files):
    roi_name = jf.replace(".json", "")
    image_key = roi_name + ".png"
    with open(os.path.join(JSON_DIR, jf)) as fh:
        data = json.load(fh)
    for cid_str, cell in data["cells"].items():
        cid = int(cid_str)
        prob_map[(image_key, cid)] = {
            f"{m}_prob": cell[JSON_KEY[m]]["probs"][LYMPH_IDX[m]] for m in METHODS
        }
    if (ji + 1) % 500 == 0:
        print(f"  Read {ji+1}/{len(json_files)} JSONs...")
print(f"  Read {len(json_files)} JSONs, {len(prob_map):,} cells")

for m in METHODS:
    df_cells[f"{m}_prob"] = df_cells.apply(
        lambda r, mm=m: prob_map.get((r["image"], r["nucleus_id"]), {}).get(f"{mm}_prob", float("nan")),
        axis=1)
n_before = len(df_cells)
df_cells = df_cells.dropna(subset=[f"{m}_prob" for m in METHODS])
n_after = len(df_cells)
if n_before != n_after:
    print(f"  Warning: dropped {n_before - n_after} cells without all probs")

df_cells[["image", "nucleus_id", "tissue_region", "gt_lymphocyte"] +
         [f"{m}_prob" for m in METHODS]].to_csv(
    os.path.join(OUT_DIR, "pr_scores_v16.csv"), index=False)

print("Bootstrapping AP/AUC (Global + per-tissue, ROI-level resampling)...")
ap_auc_rows = []
gt_bool = df_cells["gt_lymphocyte"].astype(int).values
row_global = {"tissue_type": 0, "tissue_name": "Global (weighted)",
              "n_cells": len(df_cells), "n_pos": int(gt_bool.sum())}
for m in METHODS:
    ap_ci = bootstrap_ci_apauc(df_cells, f"{m}_prob", average_precision_score,
                                seed=RNG_SEED + 1000 + 2 * METHODS.index(m))
    auc_ci = bootstrap_ci_apauc(df_cells, f"{m}_prob", roc_auc_score,
                                 seed=RNG_SEED + 1001 + 2 * METHODS.index(m))
    row_global[f"{m}_ap"] = ap_ci["point"]
    row_global[f"{m}_ap_ci_low"] = ap_ci["ci_low"]
    row_global[f"{m}_ap_ci_high"] = ap_ci["ci_high"]
    row_global[f"{m}_auc"] = auc_ci["point"]
    row_global[f"{m}_auc_ci_low"] = auc_ci["ci_low"]
    row_global[f"{m}_auc_ci_high"] = auc_ci["ci_high"]
ap_auc_rows.append(row_global)
for tissue in sorted(df_cells["tissue_region"].unique()):
    sub = df_cells[df_cells["tissue_region"] == tissue]
    n_pos = int(sub["gt_lymphocyte"].astype(int).sum())
    row_t = {"tissue_type": tissue, "tissue_name": TISSUE_NAMES.get(tissue, f"T{tissue}"),
             "n_cells": len(sub), "n_pos": n_pos}
    for m in METHODS:
        ap_ci = bootstrap_ci_apauc(sub, f"{m}_prob", average_precision_score,
                                    seed=RNG_SEED + tissue * 100 + 2 * METHODS.index(m))
        auc_ci = bootstrap_ci_apauc(sub, f"{m}_prob", roc_auc_score,
                                     seed=RNG_SEED + tissue * 100 + 1 + 2 * METHODS.index(m))
        row_t[f"{m}_ap"] = ap_ci["point"]
        row_t[f"{m}_ap_ci_low"] = ap_ci["ci_low"]
        row_t[f"{m}_ap_ci_high"] = ap_ci["ci_high"]
        row_t[f"{m}_auc"] = auc_ci["point"]
        row_t[f"{m}_auc_ci_low"] = auc_ci["ci_low"]
        row_t[f"{m}_auc_ci_high"] = auc_ci["ci_high"]
    ap_auc_rows.append(row_t)
df_ap_auc = pd.DataFrame(ap_auc_rows)
# Split into ap and auc files (each carries CI columns)
ap_cols = ["tissue_type", "tissue_name", "n_cells", "n_pos"] + sum(
    [[f"{m}_ap", f"{m}_ap_ci_low", f"{m}_ap_ci_high"] for m in METHODS], [])
auc_cols = ["tissue_type", "tissue_name", "n_cells", "n_pos"] + sum(
    [[f"{m}_auc", f"{m}_auc_ci_low", f"{m}_auc_ci_high"] for m in METHODS], [])
df_ap_auc[ap_cols].to_csv(os.path.join(OUT_DIR, "ap_by_tissue.csv"), index=False)
df_ap_auc[auc_cols].to_csv(os.path.join(OUT_DIR, "auc_by_tissue.csv"), index=False)


# ── §15-16. Wilcoxon (ROI-level + patient-level) ─────────────────────────────
print("Computing Wilcoxon paired tests...")
WILCOXON_PAIRS = [
    ("panoptils", "two-sided"),  # direction unclear
    ("v4", "less"),                # LOKI < v4 expected
    ("lizard", "less"),            # LOKI < lizard expected
]

def run_wilcoxon_block(df, label):
    rows = []
    err_loki = abs_errors(df, LOKI)
    for other, alt in WILCOXON_PAIRS:
        err_other = abs_errors(df, other)
        res = paired_wilcoxon(err_loki, err_other, alt)
        rows.append(dict(comparison=f"{LOKI}_vs_{other}", level=label, **res))
    return rows

df_wilcoxon_roi = pd.DataFrame(run_wilcoxon_block(df_roi, "roi"))
df_wilcoxon_roi.to_csv(os.path.join(OUT_DIR, "wilcoxon_roi.csv"), index=False)
df_wilcoxon_pat = pd.DataFrame(run_wilcoxon_block(df_patient, "patient"))
df_wilcoxon_pat.to_csv(os.path.join(OUT_DIR, "wilcoxon_patient.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Generate docs/eval_tables_summary.md
# ═══════════════════════════════════════════════════════════════════════════════
def fmt(v, decimals=4):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "—" if np.isnan(v) else "inf"
    return f"{v:.{decimals}f}"


def fmt_ci(point, lo, hi, decimals=2):
    if np.isnan(point):
        return "—"
    return f"{fmt(point, decimals)} [{fmt(lo, decimals)}, {fmt(hi, decimals)}]"


def bold_best(vals, higher_better=True):
    valid = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float)) and np.isfinite(v)]
    if not valid:
        return [fmt(v) for v in vals]
    best_idx = (max(valid, key=lambda x: x[1])[0] if higher_better
                else min(valid, key=lambda x: x[1])[0])
    out = []
    for i, v in enumerate(vals):
        s = fmt(v)
        if i == best_idx:
            s = f"**{s}**"
        out.append(s)
    return out


lines = []
L = lines.append

L("# TCGA Holdout Ablation v2: Lizard / PanopTILs / Context-soft / Loki-OT")
L("")
L(f"Count-based metrics on {n_rois} TCGA holdout ROIs ({n_nuclei:,} nuclei, {n_patients} patients).")
L("Bootstrap 95% CI (N={}, seed={}); paired Wilcoxon at ROI and patient level (Loki-OT-anchored).".format(N_BOOT, RNG_SEED))
L("")
L("**GT canonical:** TP+FN (matches F1/P/R basis; per-tissue gt_mapped not available in csv layout).")
L("")
L("## Methods")
L("")
L("| Key | Description |")
L("|---|---|")
L("| Lizard | CellViT++ MLP head, general domain (Lizard dataset) |")
L("| PanopTILs | CellViT++ MLP head, breast-specific (151 TCGA-BRCA patients, 800K cells) |")
L("| Context-soft (v4) | Stage 1 only — context-enhanced MLP, MLLM density supervision |")
L("| Loki-OT (v15) | Stage 2 — UOT-distilled student (the deployable model) |")
L("")

# §1
L("## 1. Raw Cell Counts (GT vs Predicted Lymphocytes)")
L("")
L("| Tissue | Total Nuclei | GT Lymph | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
L("|---|---|---|" + "---|" * len(METHODS))
for _, row in df_counts.iterrows():
    cells = [str(row['total_nuclei']), str(row['gt_lymph'])] + [str(row[f"{m}_pred"]) for m in METHODS]
    L(f"| {row['tissue_name']} | " + " | ".join(cells) + " |")
L("")

# §2 P/R/F1
for label, key in [("Precision", "prec"), ("Recall", "rec"), ("F1 Score", "f1")]:
    L(f"## 2{key.upper()[0]}. {label} by Tissue")
    L("")
    L("| Tissue | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
    L("|---|" + "---|" * len(METHODS))
    for _, row in df_prf.iterrows():
        vals = [row[f"{m}_{key}"] for m in METHODS]
        fmted = bold_best(vals, higher_better=True)
        L(f"| {row['tissue_name']} | " + " | ".join(fmted) + " |")
    L("")

# §3 FP/TP/Sel
for metric_label, fp_col, higher in [("FP Elimination (%FP Removed vs Lizard)", "fp_removed_pct", True),
                                       ("TP Loss (%TP Lost vs Lizard)", "tp_lost_pct", False),
                                       ("Selectivity (FP↓% / TP↓%)", "selectivity", True)]:
    L(f"## 3. {metric_label}")
    L("")
    L("| Tissue | " + " | ".join(METHOD_LABELS[m] for m in non_baseline) + " |")
    L("|---|" + "---|" * len(non_baseline))
    for tissue in [0] + list(range(1, 8)):
        name = "Global" if tissue == 0 else TISSUE_NAMES[tissue]
        vals = []
        for m in non_baseline:
            sub = df_fp[(df_fp["tissue"] == tissue) & (df_fp["version"] == m)]
            vals.append(sub[fp_col].values[0] if len(sub) > 0 else float("nan"))
        if fp_col == "selectivity":
            fmted = []
            valid_finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
            for i, v in enumerate(vals):
                if np.isinf(v):
                    fmted.append("inf")
                elif np.isnan(v):
                    fmted.append("—")
                else:
                    s = fmt(v, 2)
                    if valid_finite and max(valid_finite, key=lambda x: x[1])[0] == i:
                        s = f"**{s}**"
                    fmted.append(s)
        else:
            fmted = bold_best(vals, higher_better=higher)
        L(f"| {name} | " + " | ".join(fmted) + " |")
    L("")

# §4 TIL density
L("## 4. TIL Density (Predicted Lymphocyte %)")
L("")
L("| Tissue | GT% | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
L("|---|---|" + "---|" * len(METHODS))
for _, row in df_dens.iterrows():
    gt = row["gt_density"]
    vals = [row[f"{m}_density"] for m in METHODS]
    errs = [abs(v - gt) for v in vals]
    best_idx = int(np.argmin(errs))
    fmted = []
    for i, v in enumerate(vals):
        s = fmt(v, 2)
        if i == best_idx:
            s = f"**{s}**"
        fmted.append(s)
    L(f"| {row['tissue_name']} | {fmt(gt, 2)} | " + " | ".join(fmted) + " |")
L("")

# §5 Per-ROI MAE + CI
L("## 5. Per-ROI MAE with 95% Bootstrap CI (Global)")
L("")
L("| Method | MAE [95% CI] |")
L("|---|---|")
for _, r in df_mae_roi.iterrows():
    L(f"| {r['label']} | {fmt_ci(r['point'], r['ci_low'], r['ci_high'])} |")
L("")

# §6 Per-ROI metrics + CI
L("## 6. Per-ROI Pooled F1 / Precision / Recall with 95% Bootstrap CI")
L("")
L("| Method | F1 [95% CI] | Precision [95% CI] | Recall [95% CI] |")
L("|---|---|---|---|")
for m in METHODS:
    sub = df_metrics_roi[df_metrics_roi["method"] == m]
    f1r = sub[sub["metric"] == "f1"].iloc[0]
    pr = sub[sub["metric"] == "precision"].iloc[0]
    rr = sub[sub["metric"] == "recall"].iloc[0]
    L(f"| {METHOD_LABELS[m]} | {fmt_ci(f1r['point'], f1r['ci_low'], f1r['ci_high'], 4)} | "
      f"{fmt_ci(pr['point'], pr['ci_low'], pr['ci_high'], 4)} | "
      f"{fmt_ci(rr['point'], rr['ci_low'], rr['ci_high'], 4)} |")
L("")

# §7 Per-tissue MAE per-ROI
L("## 7. Per-Tissue MAE with 95% Bootstrap CI (Per-ROI)")
L("")
L("| Tissue | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
L("|---|" + "---|" * len(METHODS))
for tissue in sorted(df_mae_t["tissue"].unique()):
    name = TISSUE_NAMES.get(tissue, f"T{tissue}")
    cells = []
    points = []
    for m in METHODS:
        rec = df_mae_t[(df_mae_t["tissue"] == tissue) & (df_mae_t["method"] == m)].iloc[0]
        cells.append(fmt_ci(rec['point'], rec['ci_low'], rec['ci_high']))
        points.append(rec['point'])
    best_idx = int(np.nanargmin(points))
    cells[best_idx] = f"**{cells[best_idx]}**"
    L(f"| {name} | " + " | ".join(cells) + " |")
L("")

# §8 Pearson r per tissue
L("## 8. Per-Tissue Pearson r (point estimate)")
L("")
L("| Tissue | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
L("|---|" + "---|" * len(METHODS))
for _, row in df_pearson_t.iterrows():
    vals = [row[f"{m}_r"] for m in METHODS]
    fmted = bold_best(vals, higher_better=True)
    L(f"| {row['tissue_name']} | " + " | ".join(fmted) + " |")
L("")

# §10 Per-patient MAE
L(f"## 10. Per-Patient MAE with 95% Bootstrap CI (Global, n={n_patients})")
L("")
L("| Method | MAE [95% CI] |")
L("|---|---|")
for _, r in df_mae_pat.iterrows():
    L(f"| {r['label']} | {fmt_ci(r['point'], r['ci_low'], r['ci_high'])} |")
L("")

# §11 Per-patient metrics
L(f"## 11. Per-Patient Pooled F1 / Precision / Recall with 95% Bootstrap CI (n={n_patients})")
L("")
L("| Method | F1 [95% CI] | Precision [95% CI] | Recall [95% CI] |")
L("|---|---|---|---|")
for m in METHODS:
    sub = df_metrics_pat[df_metrics_pat["method"] == m]
    f1r = sub[sub["metric"] == "f1"].iloc[0]
    pr = sub[sub["metric"] == "precision"].iloc[0]
    rr = sub[sub["metric"] == "recall"].iloc[0]
    L(f"| {METHOD_LABELS[m]} | {fmt_ci(f1r['point'], f1r['ci_low'], f1r['ci_high'], 4)} | "
      f"{fmt_ci(pr['point'], pr['ci_low'], pr['ci_high'], 4)} | "
      f"{fmt_ci(rr['point'], rr['ci_low'], rr['ci_high'], 4)} |")
L("")

# §11b Per-tissue MAE per-patient
L(f"## 11b. Per-Tissue MAE with 95% Bootstrap CI (Per-Patient, ≥{PATIENT_TISSUE_MIN} patients)")
L("")
L("| Tissue | n_patients | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
L("|---|---|" + "---|" * len(METHODS))
for tissue in sorted(df_mae_t_pat["tissue"].unique()):
    name = TISSUE_NAMES.get(tissue, f"T{tissue}")
    sub_t = df_mae_t_pat[df_mae_t_pat["tissue"] == tissue]
    n_pat = int(sub_t.iloc[0]["n"])
    cells = []
    points = []
    skipped = sub_t.iloc[0].get("note", "")
    if skipped and "skipped" in str(skipped):
        L(f"| {name} | {n_pat} | _{skipped}_ |")
        continue
    for m in METHODS:
        rec = sub_t[sub_t["method"] == m].iloc[0]
        cells.append(fmt_ci(rec['point'], rec['ci_low'], rec['ci_high']))
        points.append(rec['point'])
    if not all(np.isnan(p) for p in points):
        best_idx = int(np.nanargmin(points))
        cells[best_idx] = f"**{cells[best_idx]}**"
    L(f"| {name} | {n_pat} | " + " | ".join(cells) + " |")
L("")

# §13 AP
L("## 13. Average Precision (AP) by Tissue (from per-cell probs)")
L("")
L("| Tissue | n_cells | n_pos | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
L("|---|---|---|" + "---|" * len(METHODS))
for _, row in df_ap_auc.iterrows():
    vals = [row[f"{m}_ap"] for m in METHODS]
    fmted = bold_best(vals, higher_better=True)
    L(f"| {row['tissue_name']} | {row['n_cells']} | {row['n_pos']} | " + " | ".join(fmted) + " |")
L("")

# §14 AUC
L("## 14. ROC-AUC by Tissue (from per-cell probs)")
L("")
L("| Tissue | n_cells | n_pos | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
L("|---|---|---|" + "---|" * len(METHODS))
for _, row in df_ap_auc.iterrows():
    vals = [row[f"{m}_auc"] for m in METHODS]
    fmted = bold_best(vals, higher_better=True)
    L(f"| {row['tissue_name']} | {row['n_cells']} | {row['n_pos']} | " + " | ".join(fmted) + " |")
L("")

# §15-16 Wilcoxon
L("## 15. Paired Wilcoxon Tests — Loki-OT vs others")
L("")
L("Each row tests Wilcoxon signed-rank on per-row absolute errors |pred − gt|.")
L("Median diff = median(|err_LOKI| − |err_other|); negative ⇒ Loki-OT lower errors.")
L("")
L("### ROI-level (n=1741)")
L("")
L("| Comparison | Alt | stat | p-value | median diff [95% CI] |")
L("|---|---|---|---|---|")
for _, r in df_wilcoxon_roi.iterrows():
    L(f"| {r['comparison']} | {r['alternative']} | {fmt(r['stat'], 0)} | {fmt(r['p_value'], 4)} | "
      f"{fmt_ci(r['median_diff'], r['median_ci_low'], r['median_ci_high'], 3)} |")
L("")
L(f"### Patient-level (n={n_patients})")
L("")
L("| Comparison | Alt | stat | p-value | median diff [95% CI] |")
L("|---|---|---|---|---|")
for _, r in df_wilcoxon_pat.iterrows():
    L(f"| {r['comparison']} | {r['alternative']} | {fmt(r['stat'], 0)} | {fmt(r['p_value'], 4)} | "
      f"{fmt_ci(r['median_diff'], r['median_ci_low'], r['median_ci_high'], 3)} |")
L("")

# §17 Summary
L("## 17. Summary")
L("")
L("| Metric | Best Method (point estimate) | Notes |")
L("|---|---|---|")

f1_global = {m: df_prf[df_prf["tissue"] == 0].iloc[0][f"{m}_f1"] for m in METHODS}
best = max(f1_global, key=f1_global.get)
L(f"| Global Micro F1 | {METHOD_LABELS[best]} ({fmt(f1_global[best])}) | |")

macro_global = {m: df_prf[df_prf["tissue"] == -1].iloc[0][f"{m}_f1"] for m in METHODS}
best = max(macro_global, key=macro_global.get)
L(f"| Global Macro F1 | {METHOD_LABELS[best]} ({fmt(macro_global[best])}) | |")

mae_roi_map = {r["method"]: r["point"] for _, r in df_mae_roi.iterrows()}
best = min(mae_roi_map, key=mae_roi_map.get)
L(f"| Per-ROI MAE | {METHOD_LABELS[best]} ({fmt(mae_roi_map[best], 2)}) | Lower = better |")

mae_pat_map = {r["method"]: r["point"] for _, r in df_mae_pat.iterrows()}
best = min(mae_pat_map, key=mae_pat_map.get)
L(f"| Per-Patient MAE | {METHOD_LABELS[best]} ({fmt(mae_pat_map[best], 2)}) | Lower = better |")

ap_global = df_ap_auc[df_ap_auc["tissue_type"] == 0].iloc[0]
ap_map = {m: ap_global[f"{m}_ap"] for m in METHODS}
best = max(ap_map, key=ap_map.get)
L(f"| Global AP | {METHOD_LABELS[best]} ({fmt(ap_map[best])}) | Higher = better |")

auc_map = {m: ap_global[f"{m}_auc"] for m in METHODS}
best = max(auc_map, key=auc_map.get)
L(f"| Global ROC-AUC | {METHOD_LABELS[best]} ({fmt(auc_map[best])}) | Higher = better |")

L("")
L("---")
L("")
L(f"Generated by `eval/compute_metrics.py`.")

os.makedirs(os.path.join(REPO_ROOT, "docs"), exist_ok=True)
readme_path = os.path.join(REPO_ROOT, "docs", "eval_tables_summary.md")
with open(readme_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nDone. Outputs in {OUT_DIR}/")
print(f"README: {readme_path}")
