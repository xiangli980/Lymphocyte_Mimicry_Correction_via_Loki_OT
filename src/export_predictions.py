"""
Export CellViT predictions (lizard baseline + Stage 2 Loki-OT distilled) for
all cached ROIs.

Outputs:
  - Per-cell JSON predictions -> loki_ot_classifier/{roi_name}.json
  - Comparison overlay plots  -> loki_ot_classifier_overlay/{roi_name}.png

Loads checkpoints/loki_ot_v15_epoch003.pth directly — that checkpoint is
already the selected best epoch.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import scipy.io as sio
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set LOKI_OT_DATA_ROOT to wherever you've placed the TIGER WSIROIS data (see README "Dataset").
DATA_ROOT    = os.environ.get('LOKI_OT_DATA_ROOT', '/DataMount/xl260/wsirois')
BASE_DIR     = os.path.join(DATA_ROOT, 'roi-level-annotations', 'tissue-cells') + '/'
IMAGE_DIR    = os.path.join(BASE_DIR, 'selected_images/')
MAT_DIR      = os.path.join(BASE_DIR, 'selected_images_output/mat/')
MASK_DIR     = os.path.join(BASE_DIR, 'masks/')
JSON_OUT_DIR = os.path.join(BASE_DIR, 'selected_images_output/loki_ot_classifier/')
PLOT_OUT_DIR = os.path.join(BASE_DIR, 'selected_images_output/loki_ot_classifier_overlay/')

CACHE_PATH        = os.path.join(REPO_ROOT, 'outputs', 'cache', 'cached_cell_tokens.pt')
LIZARD_CHECKPOINT = os.path.join(REPO_ROOT, 'cellvit_repo', 'checkpoints', 'classifier', 'sam-h', 'lizard.pth')
STAGE2_CHECKPOINT = os.path.join(REPO_ROOT, 'checkpoints', 'loki_ot_v15_epoch003.pth')

DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 1024
LYMPH_IDX  = 2

LIZARD_LABELS = {0: 'Neutrophil', 1: 'Epithelial', 2: 'Lymphocyte',
                 3: 'Plasma', 4: 'Eosinophil', 5: 'Connective'}
LIZARD_COLORS = {
    0: (0.65, 0.33, 0.16),
    1: (1.0,  0.5,  0.0 ),
    2: (0.0,  0.7,  0.0 ),
    3: (0.6,  0.0,  0.8 ),
    4: (1.0,  0.85, 0.0 ),
    5: (0.0,  0.5,  1.0 ),
}
TISSUE_NAMES = {1: 'T1 (tumor)', 2: 'T2 (immune)', 3: 'T3', 4: 'T4 (glands)',
                5: 'T5 (bg)', 6: 'T6 (stroma)', 7: 'T7 (necrosis)'}


# ============================================================
# MODEL
# ============================================================

class LinearClassifier(nn.Module):
    def __init__(self, embed_dim=1280, hidden_dim=100, num_classes=6, drop_rate=0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ContextClassifier(nn.Module):
    def __init__(self, cell_dim=1280, context_dim=1280, hidden_dim=512,
                 num_classes=6, drop_rate=0):
        super().__init__()
        self.fc1 = nn.Linear(cell_dim + context_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, cell_tokens, context_tokens):
        x = torch.cat([cell_tokens, context_tokens], dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def unflatten_dict(d, sep='.'):
    output_dict = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = output_dict
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    return output_dict


def load_stage2_classifier(checkpoint_path, lizard_checkpoint_path):
    """Load trained Stage 2 ContextClassifier from checkpoint."""
    lizard_cp   = torch.load(lizard_checkpoint_path, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')
    hidden_dim  = lizard_conf['model'].get('hidden_dim', 100)
    num_classes = lizard_conf['data']['num_classes']

    classifier = ContextClassifier(
        cell_dim=1280, context_dim=1280,
        hidden_dim=hidden_dim, num_classes=num_classes, drop_rate=0
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    classifier.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded Stage 2 classifier from epoch {ckpt['epoch']} (loss={ckpt.get('loss', '?'):.6f})")
    return classifier


def load_lizard_classifier(lizard_checkpoint_path):
    """Load lizard baseline LinearClassifier."""
    lizard_cp   = torch.load(lizard_checkpoint_path, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')
    hidden_dim  = lizard_conf['model'].get('hidden_dim', 100)
    num_classes = lizard_conf['data']['num_classes']
    clf = LinearClassifier(embed_dim=1280, hidden_dim=hidden_dim,
                           num_classes=num_classes, drop_rate=0)
    clf.load_state_dict(lizard_cp['model_state_dict'])
    return clf


# ============================================================
# HELPERS
# ============================================================

def compute_crop_offsets(image_path, patch_size):
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w = img.shape[:2]
    sh = max(0, (h - patch_size) // 2) if h > patch_size else 0
    sw = max(0, (w - patch_size) // 2) if w > patch_size else 0
    return sh, sw


def load_and_crop(path, patch_size, sh, sw, is_mask=False):
    if is_mask:
        arr = np.array(Image.open(path))
    else:
        arr = np.array(Image.open(path).convert('RGB'))
    h, w = arr.shape[:2]
    if is_mask:
        if h < patch_size or w < patch_size:
            arr = np.pad(arr, ((0, max(0, patch_size-h)), (0, max(0, patch_size-w))),
                         constant_values=0)
        h, w = arr.shape[:2]
        if h > patch_size or w > patch_size:
            arr = arr[sh:sh+patch_size, sw:sw+patch_size]
    else:
        if h < patch_size or w < patch_size:
            arr = np.pad(arr, ((0, max(0, patch_size-h)), (0, max(0, patch_size-w)), (0, 0)),
                         mode='constant', constant_values=255)
        h, w = arr.shape[:2]
        if h > patch_size or w > patch_size:
            arr = arr[sh:sh+patch_size, sw:sw+patch_size]
    return arr


def get_cell_ids_from_instmap(inst_map):
    return [int(cid) for cid in np.unique(inst_map)
            if cid != 0 and (inst_map == cid).sum() >= 10]


def get_cell_centroids(inst_map, cell_ids):
    centroids = {}
    for cid in cell_ids:
        ys, xs = np.where(inst_map == cid)
        centroids[cid] = [float(ys.mean()), float(xs.mean())]
    return centroids


# ============================================================
# PLOT
# ============================================================

def make_comparison_plot(roi_name, img, inst_map, cell_ids,
                         liz_classes, stage2_classes, output_path):
    """2-row comparison: Lizard (top) vs Stage 2 Loki-OT distilled (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 15))
    fig.suptitle(f'{roi_name}', fontsize=14, fontweight='bold')

    for row, (name, classes) in enumerate([
        ('Lizard (baseline)', liz_classes),
        ('Stage 2 (Loki-OT distilled)', stage2_classes),
    ]):
        # Col 0: cell type overlay
        overlay    = img.copy().astype(float) / 255.0
        type_rgb   = np.zeros_like(overlay)
        for i, cid in enumerate(cell_ids):
            mask = inst_map == cid
            type_rgb[mask] = LIZARD_COLORS[classes[i]]
        blended = np.clip(0.6 * overlay + 0.4 * type_rgb, 0, 1)
        axes[row, 0].imshow(blended)
        axes[row, 0].set_title(f'{name}\nCell Type Overlay', fontsize=12)
        axes[row, 0].axis('off')

        # Col 1: lymphocyte highlight
        lymph_overlay = img.copy().astype(float) / 255.0 * 0.4
        for i, cid in enumerate(cell_ids):
            mask = inst_map == cid
            if classes[i] == LYMPH_IDX:
                lymph_overlay[mask] = [0.0, 1.0, 0.0]
            else:
                lymph_overlay[mask] = [0.5, 0.5, 0.5]
        n_lymph   = sum(1 for c in classes if c == LYMPH_IDX)
        pct_lymph = n_lymph / len(classes) * 100 if classes else 0
        axes[row, 1].imshow(lymph_overlay)
        axes[row, 1].set_title(f'Lymphocytes: {n_lymph}/{len(classes)} ({pct_lymph:.1f}%)',
                                fontsize=12)
        axes[row, 1].axis('off')

        # Col 2: class distribution bar
        counts = [0] * 6
        for c in classes:
            counts[c] += 1
        bars = axes[row, 2].bar(
            range(6), counts,
            color=[LIZARD_COLORS[i] for i in range(6)],
            edgecolor='black', alpha=0.85
        )
        axes[row, 2].set_xticks(range(6))
        axes[row, 2].set_xticklabels([LIZARD_LABELS[i] for i in range(6)],
                                      rotation=30, ha='right', fontsize=9)
        axes[row, 2].set_ylabel('Count')
        axes[row, 2].set_title(f'{name} Distribution', fontsize=12)
        axes[row, 2].grid(axis='y', alpha=0.3)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                pct = cnt / len(classes) * 100
                axes[row, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                                  f'{cnt}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(JSON_OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_OUT_DIR, exist_ok=True)

    # Load models
    print(f"Loading Stage 2 classifier checkpoint: {STAGE2_CHECKPOINT}")
    stage2_clf = load_stage2_classifier(STAGE2_CHECKPOINT, LIZARD_CHECKPOINT)
    stage2_clf.eval().to(DEVICE)

    print(f"Loading lizard baseline classifier...")
    liz_clf = load_lizard_classifier(LIZARD_CHECKPOINT)
    liz_clf.eval().to(DEVICE)

    print(f"\nLoading cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    roi_names = sorted(cache.keys())
    print(f"  {len(roi_names)} ROIs loaded")

    print(f"\nExporting JSON + plots for all {len(roi_names)} ROIs...")
    print(f"  {'ROI':<45s}, {'cells':>5s}, {'liz_lymph%':>10s}, {'s2_lymph%':>10s}, {'change':>8s}")

    n_json  = 0
    n_plots = 0

    for roi_name in roi_names:
        roi         = cache[roi_name]
        lizard_soft = roi['lizard_soft'].float()
        cell_tissue = roi['cell_tissue']
        n_cells     = lizard_soft.shape[0]

        # Forward passes
        with torch.no_grad():
            cell_t = roi['cell_tokens'].float().to(DEVICE)
            ctx_t  = roi['context_tokens'].float().to(DEVICE)

            liz_logits    = liz_clf(cell_t)
            stage2_logits = stage2_clf(cell_t, ctx_t)

            liz_probs    = F.softmax(liz_logits, dim=1).cpu()
            stage2_probs = F.softmax(stage2_logits, dim=1).cpu()

        liz_classes    = liz_probs.argmax(dim=1).tolist()
        stage2_classes = stage2_probs.argmax(dim=1).tolist()

        liz_counts    = [0] * 6
        stage2_counts = [0] * 6
        for i in range(n_cells):
            liz_counts[liz_classes[i]]       += 1
            stage2_counts[stage2_classes[i]] += 1

        liz_lymph_pct    = liz_counts[LYMPH_IDX]    / n_cells if n_cells > 0 else 0
        stage2_lymph_pct = stage2_counts[LYMPH_IDX] / n_cells if n_cells > 0 else 0
        change = stage2_lymph_pct - liz_lymph_pct

        # Load inst_map and compute crop offsets
        mat_path = os.path.join(MAT_DIR, f'{roi_name}.mat')
        img_path = os.path.join(IMAGE_DIR, f'{roi_name}.png')

        if not os.path.exists(mat_path) or not os.path.exists(img_path):
            print(f"  WARN: missing mat or image for {roi_name}, skipping")
            continue

        sh, sw = compute_crop_offsets(img_path, PATCH_SIZE)

        mat_data     = sio.loadmat(mat_path)
        inst_map_raw = mat_data['inst_map']
        h, w = inst_map_raw.shape[:2]
        if h < PATCH_SIZE or w < PATCH_SIZE:
            inst_map_raw = np.pad(inst_map_raw,
                                  ((0, max(0, PATCH_SIZE-h)), (0, max(0, PATCH_SIZE-w))),
                                  constant_values=0)
            h, w = inst_map_raw.shape[:2]
        if h > PATCH_SIZE or w > PATCH_SIZE:
            inst_map_raw = inst_map_raw[sh:sh+PATCH_SIZE, sw:sw+PATCH_SIZE]
        inst_map = inst_map_raw

        cell_ids = get_cell_ids_from_instmap(inst_map)
        if len(cell_ids) != n_cells:
            print(f"  WARN: {roi_name} inst_map cell count ({len(cell_ids)}) != cache ({n_cells}), skipping")
            continue

        centroids = get_cell_centroids(inst_map, cell_ids)

        # Build cells dict
        cells_dict = {}
        for i, cid in enumerate(cell_ids):
            tissue_t = int(cell_tissue[i].item())
            cells_dict[str(cid)] = {
                'centroid':   centroids[cid],
                'tissue_type': tissue_t,
                'lizard_baseline': {
                    'type':  liz_classes[i],
                    'probs': [round(float(p), 4) for p in liz_probs[i].tolist()],
                },
                'stage2_loki_ot': {
                    'type':           stage2_classes[i],
                    'probs':          [round(float(p), 4) for p in stage2_probs[i].tolist()],
                    'reclassified':   liz_classes[i] != stage2_classes[i],
                },
            }

        # Write JSON
        output_json = {
            'roi_name':  roi_name,
            'num_cells': n_cells,
            'method':    'loki_ot_selective_ot_distilled_stage1_fallback_context_classifier',
            'classifiers': {
                'lizard_baseline': {
                    'label_map': {str(k): v for k, v in LIZARD_LABELS.items()},
                    'summary': {'lymphocyte_pct': round(liz_lymph_pct, 4), 'counts': liz_counts},
                },
                'stage2_loki_ot': {
                    'label_map': {str(k): v for k, v in LIZARD_LABELS.items()},
                    'summary': {'lymphocyte_pct': round(stage2_lymph_pct, 4), 'counts': stage2_counts},
                },
            },
            'cells': cells_dict,
        }

        json_path = os.path.join(JSON_OUT_DIR, f'{roi_name}.json')
        with open(json_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        n_json += 1

        # Write plot
        img = load_and_crop(img_path, PATCH_SIZE, sh, sw, is_mask=False)
        plot_path = os.path.join(PLOT_OUT_DIR, f'{roi_name}.png')
        make_comparison_plot(roi_name, img, inst_map, cell_ids,
                             liz_classes, stage2_classes, plot_path)
        n_plots += 1

        print(f"  {roi_name:<45s}, {n_cells:5d}, {liz_lymph_pct:9.1%}, {stage2_lymph_pct:9.1%}, {change:+7.1%}")

    print(f"\nDone.")
    print(f"  JSON  ({n_json}  files): {JSON_OUT_DIR}")
    print(f"  Plots ({n_plots} files): {PLOT_OUT_DIR}")


if __name__ == '__main__':
    main()
