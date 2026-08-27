"""
TCGA Classification: lizard / panoptils / Stage 1 / Stage 2 on TCGA test ROIs
using cached CellViT embeddings (produced by extract_tcga_embeddings.py).

Adds PanopTILs (4-class breast-pretrained MLP) as a baseline alongside lizard.
PanopTILs classes: {Other Cells, Epithelial, Stromal, TILs}
TILs (class 3) is the lymphocyte equivalent for comparison.

Input:  Cached .pt embeddings from extract_tcga_embeddings.py
Output: JSON per ROI + 4-row comparison plots (lizard / panoptils / Stage 1 / Stage 2)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
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

# External data — must match extract_tcga_embeddings.py's paths.
# Set LOKI_OT_DATA_ROOT to wherever you've placed the TIGER WSIROIS data (see README "Dataset").
DATA_ROOT     = os.environ.get('LOKI_OT_DATA_ROOT', '/DataMount/xl260/wsirois')
BASE_DIR      = os.path.join(DATA_ROOT, 'roi-level-annotations', 'tissue-cells') + '/'
ROI_IMAGE_DIR = os.path.join(BASE_DIR, 'images/')
MAT_DIR       = os.path.join(BASE_DIR, 'selected_tcga_stardist_mat/')

# Input: cached CellViT embeddings, written by extract_tcga_embeddings.py
EMBED_DIR     = os.path.join(BASE_DIR, 'selected_images_output/cellvit_tcga_embeddings/')

# Output
BASE_OUT      = os.path.join(BASE_DIR, 'selected_images_output/')
JSON_OUT_DIR  = os.path.join(BASE_OUT, 'loki_ot_classifier_tcga/')
PLOT_OUT_DIR  = os.path.join(BASE_OUT, 'loki_ot_classifier_tcga_overlay/')

# Model checkpoints
LIZARD_CHECKPOINT    = os.path.join(REPO_ROOT, 'cellvit_repo', 'checkpoints', 'classifier', 'sam-h', 'lizard.pth')
PANOPTILS_CHECKPOINT = os.path.join(REPO_ROOT, 'cellvit_repo', 'checkpoints', 'classifier', 'sam-h', 'panoptils.pth')
STAGE1_CHECKPOINT    = os.path.join(REPO_ROOT, 'checkpoints', 'context_soft_v4_epoch018.pth')
STAGE2_CHECKPOINT    = os.path.join(REPO_ROOT, 'checkpoints', 'loki_ot_v15_epoch003.pth')

DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Lizard: 6-class, lymphocyte = class 2
LYMPH_IDX = 2
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

# PanopTILs: 4-class, TILs = class 3
PANOPTILS_TIL_IDX = 3
PANOPTILS_LABELS = {0: 'Other Cells', 1: 'Epithelial', 2: 'Stromal', 3: 'TILs'}
PANOPTILS_COLORS = {
    0: (0.5,  0.5,  0.5 ),   # gray - Other
    1: (1.0,  0.5,  0.0 ),   # orange - Epithelial
    2: (0.0,  0.5,  1.0 ),   # blue - Stromal
    3: (0.0,  0.7,  0.0 ),   # green - TILs (same as lymphocyte)
}


# ============================================================
# MODELS
# ============================================================

class LinearClassifier(nn.Module):
    def __init__(self, embed_dim=1280, hidden_dim=512, num_classes=6, drop_rate=0):
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


# ============================================================
# MODEL LOADING
# ============================================================

def load_lizard_classifier():
    lizard_cp   = torch.load(LIZARD_CHECKPOINT, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')
    hidden_dim  = lizard_conf['model'].get('hidden_dim', 100)
    num_classes = lizard_conf['data']['num_classes']
    clf = LinearClassifier(embed_dim=1280, hidden_dim=hidden_dim,
                           num_classes=num_classes, drop_rate=0)
    clf.load_state_dict(lizard_cp['model_state_dict'])
    clf.eval().to(DEVICE)
    print(f"  Loaded lizard classifier (hidden_dim={hidden_dim}, num_classes={num_classes})")
    return clf


def load_panoptils_classifier():
    cp   = torch.load(PANOPTILS_CHECKPOINT, map_location='cpu', weights_only=False)
    conf = unflatten_dict(cp['config'], '.')
    hidden_dim  = conf['model'].get('hidden_dim', 512)
    num_classes = conf['data']['num_classes']
    clf = LinearClassifier(embed_dim=1280, hidden_dim=hidden_dim,
                           num_classes=num_classes, drop_rate=0)
    clf.load_state_dict(cp['model_state_dict'])
    clf.eval().to(DEVICE)
    print(f"  Loaded panoptils classifier (hidden_dim={hidden_dim}, num_classes={num_classes})")
    return clf


def load_context_classifier(checkpoint_path, name):
    lizard_cp   = torch.load(LIZARD_CHECKPOINT, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')
    hidden_dim  = lizard_conf['model'].get('hidden_dim', 100)
    num_classes = lizard_conf['data']['num_classes']

    clf = ContextClassifier(
        cell_dim=1280, context_dim=1280,
        hidden_dim=hidden_dim, num_classes=num_classes, drop_rate=0
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    clf.load_state_dict(ckpt['model_state_dict'])
    clf.eval().to(DEVICE)
    epoch = ckpt.get('epoch', '?')
    loss  = ckpt.get('loss', None)
    loss_str = f", loss={loss:.6f}" if loss is not None else ""
    print(f"  Loaded {name} classifier (epoch={epoch}{loss_str})")
    return clf


# ============================================================
# ROI PARSING
# ============================================================

def parse_roi_name(roi_name):
    m = re.match(r'^(.+?)_\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]$', roi_name)
    if not m:
        return None, None
    prefix = m.group(1)
    bbox = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return prefix, bbox


# ============================================================
# PLOT (4-row: lizard / panoptils / Stage 1 / Stage 2)
# ============================================================

def make_comparison_plot(roi_name, img, inst_map, cell_ids,
                         liz_classes, panop_classes, stage1_classes, stage2_classes,
                         output_path):
    """4-row comparison: Lizard / PanopTILs / Stage 1 / Stage 2."""
    img_h, img_w = img.shape[:2]
    im_h, im_w = inst_map.shape[:2]
    if (im_h, im_w) != (img_h, img_w):
        inst_map = np.array(Image.fromarray(inst_map.astype(np.int32)).resize(
            (img_w, img_h), Image.NEAREST))

    fig, axes = plt.subplots(4, 3, figsize=(22, 29))
    fig.suptitle(f'{roi_name}', fontsize=14, fontweight='bold')

    rows = [
        ('Lizard (baseline)',       liz_classes,    LIZARD_LABELS,    LIZARD_COLORS,    LYMPH_IDX,         6),
        ('PanopTILs (breast)',      panop_classes,  PANOPTILS_LABELS, PANOPTILS_COLORS, PANOPTILS_TIL_IDX, 4),
        ('Stage 1 (context_soft)',  stage1_classes, LIZARD_LABELS,    LIZARD_COLORS,    LYMPH_IDX,         6),
        ('Stage 2 (Loki-OT)',       stage2_classes, LIZARD_LABELS,    LIZARD_COLORS,    LYMPH_IDX,         6),
    ]

    for row, (name, classes, labels, colors, til_idx, n_cls) in enumerate(rows):
        # Col 0: cell type overlay
        overlay  = img.copy().astype(float) / 255.0
        type_rgb = np.zeros_like(overlay)
        for i, cid in enumerate(cell_ids):
            mask = inst_map == cid
            type_rgb[mask] = colors[classes[i]]
        blended = np.clip(0.6 * overlay + 0.4 * type_rgb, 0, 1)
        axes[row, 0].imshow(blended)
        axes[row, 0].set_title(f'{name}\nCell Type Overlay', fontsize=12)
        axes[row, 0].axis('off')

        # Col 1: lymphocyte/TILs highlight
        lymph_overlay = img.copy().astype(float) / 255.0 * 0.4
        for i, cid in enumerate(cell_ids):
            mask = inst_map == cid
            if classes[i] == til_idx:
                lymph_overlay[mask] = [0.0, 1.0, 0.0]
            else:
                lymph_overlay[mask] = [0.5, 0.5, 0.5]
        n_lymph   = sum(1 for c in classes if c == til_idx)
        pct_lymph = n_lymph / len(classes) * 100 if classes else 0
        til_label = 'TILs' if til_idx == PANOPTILS_TIL_IDX and n_cls == 4 else 'Lymphocytes'
        axes[row, 1].imshow(lymph_overlay)
        axes[row, 1].set_title(f'{til_label}: {n_lymph}/{len(classes)} ({pct_lymph:.1f}%)',
                                fontsize=12)
        axes[row, 1].axis('off')

        # Col 2: class distribution bar
        counts = [0] * n_cls
        for c in classes:
            counts[c] += 1
        bars = axes[row, 2].bar(
            range(n_cls), counts,
            color=[colors[i] for i in range(n_cls)],
            edgecolor='black', alpha=0.85
        )
        axes[row, 2].set_xticks(range(n_cls))
        axes[row, 2].set_xticklabels([labels[i] for i in range(n_cls)],
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

    # Load classifiers
    print("Loading classifiers...")
    liz_clf    = load_lizard_classifier()
    panop_clf  = load_panoptils_classifier()
    stage1_clf = load_context_classifier(STAGE1_CHECKPOINT, 'Stage 1')
    stage2_clf = load_context_classifier(STAGE2_CHECKPOINT, 'Stage 2')

    # Discover cached embeddings
    embed_files = sorted(f for f in os.listdir(EMBED_DIR) if f.endswith('.pt'))
    print(f"\n{len(embed_files)} cached embeddings found in {EMBED_DIR}")

    print(f"\n{'='*90}")
    print(f"Classifying {len(embed_files)} TCGA ROIs (lizard / panoptils / Stage 1 / Stage 2)")
    print(f"{'='*90}")
    print(f"  {'ROI':<75s} {'cells':>5s} {'liz%':>6s} {'pan%':>6s} {'s1%':>6s} {'s2%':>6s}")

    n_json = 0
    n_plots = 0
    n_skipped = 0

    for idx, pt_file in enumerate(embed_files):
        roi_name = pt_file.replace('.pt', '')
        wsi_prefix, bbox = parse_roi_name(roi_name)
        if wsi_prefix is None:
            n_skipped += 1
            continue

        # Load cached embeddings
        emb = torch.load(os.path.join(EMBED_DIR, pt_file), map_location='cpu', weights_only=False)
        cell_tokens    = emb['cell_tokens']
        context_tokens = emb['context_tokens']
        cell_ids       = emb['cell_ids']
        cell_tissue    = emb['cell_tissue']
        centroids      = emb['centroids']
        n_cells = len(cell_ids)

        if n_cells == 0:
            n_skipped += 1
            continue

        # Run classifiers
        with torch.no_grad():
            ct  = cell_tokens.float().to(DEVICE)
            ctx = context_tokens.float().to(DEVICE)

            liz_probs    = F.softmax(liz_clf(ct), dim=1).cpu()
            panop_probs  = F.softmax(panop_clf(ct), dim=1).cpu()
            stage1_probs = F.softmax(stage1_clf(ct, ctx), dim=1).cpu()
            stage2_probs = F.softmax(stage2_clf(ct, ctx), dim=1).cpu()

        liz_classes    = liz_probs.argmax(dim=1).tolist()
        panop_classes  = panop_probs.argmax(dim=1).tolist()
        stage1_classes = stage1_probs.argmax(dim=1).tolist()
        stage2_classes = stage2_probs.argmax(dim=1).tolist()

        # Counts
        liz_counts    = [0] * 6
        panop_counts  = [0] * 4
        stage1_counts = [0] * 6
        stage2_counts = [0] * 6
        for i in range(n_cells):
            liz_counts[liz_classes[i]]       += 1
            panop_counts[panop_classes[i]]   += 1
            stage1_counts[stage1_classes[i]] += 1
            stage2_counts[stage2_classes[i]] += 1

        liz_lymph    = liz_counts[LYMPH_IDX]   / n_cells if n_cells > 0 else 0
        panop_tils   = panop_counts[PANOPTILS_TIL_IDX] / n_cells if n_cells > 0 else 0
        stage1_lymph = stage1_counts[LYMPH_IDX] / n_cells if n_cells > 0 else 0
        stage2_lymph = stage2_counts[LYMPH_IDX] / n_cells if n_cells > 0 else 0

        # Build cells dict
        cells_dict = {}
        for i, cid in enumerate(cell_ids):
            cells_dict[str(cid)] = {
                'centroid':    centroids[cid],
                'tissue_type': int(cell_tissue[i].item()),
                'lizard_baseline': {
                    'type':  liz_classes[i],
                    'probs': [round(float(p), 4) for p in liz_probs[i].tolist()],
                },
                'panoptils_baseline': {
                    'type':  panop_classes[i],
                    'probs': [round(float(p), 4) for p in panop_probs[i].tolist()],
                },
                'stage1_context': {
                    'type':  stage1_classes[i],
                    'probs': [round(float(p), 4) for p in stage1_probs[i].tolist()],
                },
                'stage2_loki_ot': {
                    'type':           stage2_classes[i],
                    'probs':          [round(float(p), 4) for p in stage2_probs[i].tolist()],
                    'reclassified':   liz_classes[i] != stage2_classes[i],
                },
            }

        # Write JSON
        output_json = {
            'roi_name':    roi_name,
            'wsi_prefix':  wsi_prefix,
            'bbox_wsi':    list(bbox),
            'num_cells':   n_cells,
            'classifiers': {
                'lizard_baseline': {
                    'label_map': {str(k): v for k, v in LIZARD_LABELS.items()},
                    'summary': {'lymphocyte_pct': round(liz_lymph, 4), 'counts': liz_counts},
                },
                'panoptils_baseline': {
                    'label_map': {str(k): v for k, v in PANOPTILS_LABELS.items()},
                    'summary': {'tils_pct': round(panop_tils, 4), 'counts': panop_counts},
                },
                'stage1_context': {
                    'label_map': {str(k): v for k, v in LIZARD_LABELS.items()},
                    'summary': {'lymphocyte_pct': round(stage1_lymph, 4), 'counts': stage1_counts},
                },
                'stage2_loki_ot': {
                    'label_map': {str(k): v for k, v in LIZARD_LABELS.items()},
                    'summary': {'lymphocyte_pct': round(stage2_lymph, 4), 'counts': stage2_counts},
                },
            },
            'cells': cells_dict,
        }

        json_path = os.path.join(JSON_OUT_DIR, f'{roi_name}.json')
        with open(json_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        n_json += 1

        # Plot using ROI image + stardist inst_map
        roi_img_path = os.path.join(ROI_IMAGE_DIR, f'{roi_name}.png')
        mat_path = os.path.join(MAT_DIR, f'{roi_name}.mat')

        if os.path.exists(roi_img_path) and os.path.exists(mat_path):
            roi_img = np.array(Image.open(roi_img_path).convert('RGB'))
            inst_map = sio.loadmat(mat_path)['inst_map']
            x1, y1, x2, y2 = bbox
            roi_w, roi_h = x2 - x1, y2 - y1
            inst_map = inst_map[:roi_h, :roi_w]

            plot_path = os.path.join(PLOT_OUT_DIR, f'{roi_name}.png')
            make_comparison_plot(roi_name, roi_img, inst_map, cell_ids,
                                liz_classes, panop_classes, stage1_classes, stage2_classes,
                                plot_path)
            n_plots += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(embed_files):
            print(f"  [{idx+1}/{len(embed_files)}] {roi_name:<60s} {n_cells:5d} "
                  f"{liz_lymph:5.1%} {panop_tils:5.1%} {stage1_lymph:5.1%} {stage2_lymph:5.1%}")

    print(f"\nDone.")
    print(f"  JSON  ({n_json} files):  {JSON_OUT_DIR}")
    print(f"  Plots ({n_plots} files): {PLOT_OUT_DIR}")
    if n_skipped > 0:
        print(f"  Skipped: {n_skipped}")


if __name__ == '__main__':
    main()
