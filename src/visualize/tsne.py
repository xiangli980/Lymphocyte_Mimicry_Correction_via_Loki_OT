"""
t-SNE of post-MLP hidden features for Lizard / Stage 1 / Stage 2 classifiers.

Extracts 512-dim hidden representations (after fc1+ReLU, before fc2) and
visualizes with t-SNE colored by tissue type, ROI, and predicted cell class.

Produces three figures:
  - post_mlp_tsne_train.png          (train ROIs)
  - post_mlp_tsne_test.png           (TCGA test ROIs)
  - post_mlp_tsne_train_vs_test.png  (joint embedding)

Note: the Lizard baseline only needs cell_tokens, while Stage 1/Stage 2 need
cell_tokens + context_tokens — but train_stage1_context.py's cache already
has everything needed for all three, so a single CACHE_PATH covers all
three models here.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'cellvit_repo'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================

# External data — TCGA test embeddings, produced by extract_tcga_embeddings.py.
BASE_DIR      = '/DataMount/xl260/wsirois/roi-level-annotations/tissue-cells/'
BASE_OUT      = os.path.join(BASE_DIR, 'selected_images_output/')
EMBED_DIR     = os.path.join(BASE_OUT, 'cellvit_tcga_embeddings/')

CACHE_PATH        = os.path.join(REPO_ROOT, 'outputs', 'cache', 'cached_cell_tokens.pt')
LIZARD_CHECKPOINT = os.path.join(REPO_ROOT, 'cellvit_repo', 'checkpoints', 'classifier', 'sam-h', 'lizard.pth')
STAGE1_CHECKPOINT = os.path.join(REPO_ROOT, 'checkpoints', 'context_soft_v4_epoch018.pth')
STAGE2_CHECKPOINT = os.path.join(REPO_ROOT, 'checkpoints', 'loki_ot_v15_epoch003.pth')

VIS_DIR = os.path.join(REPO_ROOT, 'outputs', 'visualizations')

DEVICE        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_CELLS     = 8000      # subsample target per figure
TSNE_PERP     = 30
TSNE_SEED     = 42
MAX_TEST_ROIS = 300       # subsample TCGA ROIs (loading many .pt files is slow)

LIZARD_LABELS = {0: 'Neutrophil', 1: 'Epithelial', 2: 'Lymphocyte',
                 3: 'Plasma', 4: 'Eosinophil', 5: 'Connective'}
TISSUE_NAMES  = {1: 'T1 Tumor', 2: 'T2 Immune', 3: 'T3', 4: 'T4 Glands',
                 5: 'T5 BG', 6: 'T6 Stroma', 7: 'T7 Necrosis', 0: 'T0 Unknown'}
LIZARD_COLORS = {
    0: '#A6553D', 1: '#FF8000', 2: '#00B300', 3: '#9900CC', 4: '#FFD900', 5: '#0080FF',
}
TISSUE_COLORS = {
    0: '#999999', 1: '#E41A1C', 2: '#377EB8', 3: '#4DAF4A',
    4: '#984EA3', 5: '#FF7F00', 6: '#A65628', 7: '#F781BF',
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
    out = {}
    for key, value in d.items():
        keys = key.split(sep)
        cur = out
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    return out


# ============================================================
# MODEL LOADING
# ============================================================

def load_lizard_classifier():
    cp = torch.load(LIZARD_CHECKPOINT, map_location='cpu', weights_only=False)
    conf = unflatten_dict(cp['config'], '.')
    hidden_dim = conf['model'].get('hidden_dim', 100)
    num_classes = conf['data']['num_classes']
    clf = LinearClassifier(embed_dim=1280, hidden_dim=hidden_dim,
                           num_classes=num_classes, drop_rate=0)
    clf.load_state_dict(cp['model_state_dict'])
    clf.eval().to(DEVICE)
    print(f"  Lizard classifier loaded (hidden_dim={hidden_dim})")
    return clf


def load_context_classifier(ckpt_path, name):
    cp = torch.load(LIZARD_CHECKPOINT, map_location='cpu', weights_only=False)
    conf = unflatten_dict(cp['config'], '.')
    hidden_dim = conf['model'].get('hidden_dim', 100)
    num_classes = conf['data']['num_classes']
    clf = ContextClassifier(cell_dim=1280, context_dim=1280,
                            hidden_dim=hidden_dim, num_classes=num_classes, drop_rate=0)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    clf.load_state_dict(ckpt['model_state_dict'])
    clf.eval().to(DEVICE)
    print(f"  {name} classifier loaded (epoch={ckpt.get('epoch', '?')})")
    return clf


# ============================================================
# DATA LOADING
# ============================================================

def load_train_data():
    """Load training ROIs from the cached .pt file."""
    print("Loading train cache...")
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)

    all_cell_tokens = []
    all_context_tokens = []
    all_tissue = []
    all_roi = []

    roi_names = sorted(cache.keys())
    for roi_name in roi_names:
        ct = cache[roi_name]['cell_tokens']      # (N, 1280)
        ctx = cache[roi_name]['context_tokens']   # (N, 1280)
        tissue = cache[roi_name]['cell_tissue']   # (N,)
        n = ct.shape[0]
        all_cell_tokens.append(ct)
        all_context_tokens.append(ctx)
        all_tissue.append(tissue)
        all_roi.extend([roi_name] * n)

    cell_tokens = torch.cat(all_cell_tokens, dim=0)
    context_tokens = torch.cat(all_context_tokens, dim=0)
    tissue = torch.cat(all_tissue, dim=0)
    roi_labels = np.array(all_roi)

    print(f"  Train: {cell_tokens.shape[0]} cells from {len(roi_names)} ROIs")
    return cell_tokens, context_tokens, tissue, roi_labels


def load_test_data():
    """Load TCGA test ROI embeddings from individual .pt files."""
    print(f"Loading test embeddings from {EMBED_DIR}...")
    pt_files = sorted([f for f in os.listdir(EMBED_DIR) if f.endswith('.pt')])
    print(f"  Found {len(pt_files)} .pt files")

    # Subsample ROIs if too many
    if len(pt_files) > MAX_TEST_ROIS:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pt_files), MAX_TEST_ROIS, replace=False)
        pt_files = [pt_files[i] for i in sorted(idx)]
        print(f"  Subsampled to {len(pt_files)} ROIs")

    all_cell_tokens = []
    all_context_tokens = []
    all_tissue = []
    all_roi = []

    for f in pt_files:
        data = torch.load(os.path.join(EMBED_DIR, f), map_location='cpu', weights_only=False)
        ct = data['cell_tokens']
        ctx = data['context_tokens']
        tissue = data['cell_tissue']
        roi_name = f.replace('.pt', '')
        n = ct.shape[0]
        all_cell_tokens.append(ct)
        all_context_tokens.append(ctx)
        all_tissue.append(tissue)
        all_roi.extend([roi_name] * n)

    cell_tokens = torch.cat(all_cell_tokens, dim=0)
    context_tokens = torch.cat(all_context_tokens, dim=0)
    tissue = torch.cat(all_tissue, dim=0)
    roi_labels = np.array(all_roi)

    print(f"  Test: {cell_tokens.shape[0]} cells from {len(pt_files)} ROIs")
    return cell_tokens, context_tokens, tissue, roi_labels


def subsample(cell_tokens, context_tokens, tissue, roi_labels, max_n):
    """Subsample cells, stratified by ROI."""
    n = cell_tokens.shape[0]
    if n <= max_n:
        return cell_tokens, context_tokens, tissue, roi_labels

    rng = np.random.RandomState(42)
    unique_rois = np.unique(roi_labels)
    per_roi = max(1, max_n // len(unique_rois))

    selected = []
    for roi in unique_rois:
        roi_idx = np.where(roi_labels == roi)[0]
        k = min(per_roi, len(roi_idx))
        selected.append(rng.choice(roi_idx, k, replace=False))
    selected = np.concatenate(selected)

    # If still under budget, fill randomly
    if len(selected) < max_n:
        remaining = np.setdiff1d(np.arange(n), selected)
        extra = rng.choice(remaining, min(max_n - len(selected), len(remaining)), replace=False)
        selected = np.concatenate([selected, extra])

    selected = np.sort(selected)
    print(f"  Subsampled {n} -> {len(selected)} cells")
    return (cell_tokens[selected], context_tokens[selected],
            tissue[selected], roi_labels[selected])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_hidden_features(liz_clf, stage1_clf, stage2_clf, cell_tokens, context_tokens):
    """Extract 512-dim post-fc1+ReLU hidden features from all 3 classifiers."""
    with torch.no_grad():
        ct = cell_tokens.float().to(DEVICE)
        ctx = context_tokens.float().to(DEVICE)

        # Lizard: fc1(cell_tokens) -> ReLU
        h_liz = F.relu(liz_clf.fc1(ct)).cpu().numpy()

        # Stage 1 / Stage 2: fc1(cat(cell, context)) -> ReLU
        x_ctx = torch.cat([ct, ctx], dim=1)
        h_stage1 = F.relu(stage1_clf.fc1(x_ctx)).cpu().numpy()
        h_stage2 = F.relu(stage2_clf.fc1(x_ctx)).cpu().numpy()

        # Also get predicted classes
        pred_liz = F.softmax(liz_clf(ct), dim=1).argmax(dim=1).cpu().numpy()
        pred_stage1 = F.softmax(stage1_clf(ct, ctx), dim=1).argmax(dim=1).cpu().numpy()
        pred_stage2 = F.softmax(stage2_clf(ct, ctx), dim=1).argmax(dim=1).cpu().numpy()

    return (h_liz, h_stage1, h_stage2), (pred_liz, pred_stage1, pred_stage2)


# ============================================================
# PLOTTING
# ============================================================

def roi_to_color(roi_labels):
    """Hash ROI names to colors for visualization."""
    unique_rois = np.unique(roi_labels)
    cmap = plt.cm.get_cmap('hsv', max(len(unique_rois), 1))
    # Deterministic shuffle so nearby ROIs don't get similar colors
    order = np.array([int(hashlib.md5(r.encode()).hexdigest()[:8], 16) for r in unique_rois])
    sorted_idx = np.argsort(order)
    roi_color_map = {}
    for i, idx in enumerate(sorted_idx):
        roi_color_map[unique_rois[idx]] = cmap(i / max(len(unique_rois) - 1, 1))
    colors = np.array([roi_color_map[r] for r in roi_labels])
    return colors


def roi_to_dataset(roi_labels):
    """Classify ROI names into dataset: 'Hospital 2' (TC_*) vs 'Hospital 1' (###B/S_*)."""
    return np.array(['Hospital 2' if r.startswith('TC') else 'Hospital 1' for r in roi_labels])


def make_figure(hiddens, preds, tissue, roi_labels, title, output_path):
    """Create 3x4 t-SNE figure (rows=models, cols=tissue/ROI/class/dataset)."""
    h_liz, h_stage1, h_stage2 = hiddens
    pred_liz, pred_stage1, pred_stage2 = preds
    tissue_np = tissue.numpy() if torch.is_tensor(tissue) else tissue
    dataset_labels = roi_to_dataset(roi_labels)

    model_names = ['Lizard (baseline)', 'Stage 1 (context_soft)', 'Stage 2 (Loki-OT)']
    model_hiddens = [h_liz, h_stage1, h_stage2]
    model_preds = [pred_liz, pred_stage1, pred_stage2]

    # Run t-SNE for each model
    print("  Running t-SNE...")
    tsne_results = []
    for i, (name, h) in enumerate(zip(model_names, model_hiddens)):
        perp = min(TSNE_PERP, h.shape[0] // 2)
        tsne = TSNE(n_components=2, random_state=TSNE_SEED, perplexity=perp)
        coords = tsne.fit_transform(h)
        tsne_results.append(coords)
        print(f"    {name}: done ({h.shape[0]} cells, {h.shape[1]}-dim)")

    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(32, 22))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    col_titles = ['Tissue Type', 'ROI Identity', 'Predicted Cell Class', 'Dataset']
    roi_colors = roi_to_color(roi_labels)

    # Legends
    unique_tissues = np.unique(tissue_np)
    tissue_legend = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=TISSUE_COLORS.get(t, '#999999'),
                            markersize=8, label=TISSUE_NAMES.get(t, f'T{t}'))
                     for t in sorted(unique_tissues)]
    class_legend = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=LIZARD_COLORS[c],
                           markersize=8, label=LIZARD_LABELS[c])
                    for c in range(6)]
    n_h2 = (dataset_labels == 'Hospital 2').sum()
    n_h1 = (dataset_labels == 'Hospital 1').sum()
    dataset_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markersize=8, label=f'Hospital 1 ({n_h1})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markersize=8, label=f'Hospital 2 ({n_h2})'),
    ]

    for row, (name, coords, pred) in enumerate(zip(model_names, tsne_results, model_preds)):
        # Col 0: tissue type
        ax = axes[row, 0]
        for t in sorted(unique_tissues):
            mask = tissue_np == t
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=TISSUE_COLORS.get(t, '#999999'), s=4, alpha=0.5,
                          rasterized=True)
        ax.set_title(f'{name}\n{col_titles[0]}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(handles=tissue_legend, loc='upper right', fontsize=7,
                     markerscale=0.8, framealpha=0.7)

        # Col 1: ROI identity
        ax = axes[row, 1]
        ax.scatter(coords[:, 0], coords[:, 1], c=roi_colors, s=4, alpha=0.5,
                  rasterized=True)
        ax.set_title(f'{name}\n{col_titles[1]}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        n_rois = len(np.unique(roi_labels))
        ax.text(0.02, 0.02, f'{n_rois} ROIs', transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom')

        # Col 2: predicted cell class
        ax = axes[row, 2]
        for c in range(6):
            mask = pred == c
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=LIZARD_COLORS[c], s=4, alpha=0.5,
                          rasterized=True)
        ax.set_title(f'{name}\n{col_titles[2]}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(handles=class_legend, loc='upper right', fontsize=7,
                     markerscale=0.8, framealpha=0.7)

        # Col 3: dataset (Hospital 1 vs Hospital 2)
        ax = axes[row, 3]
        h1_mask = dataset_labels == 'Hospital 1'
        h2_mask = dataset_labels == 'Hospital 2'
        ax.scatter(coords[h2_mask, 0], coords[h2_mask, 1],
                  c='#1f77b4', s=4, alpha=0.4, rasterized=True)
        ax.scatter(coords[h1_mask, 0], coords[h1_mask, 1],
                  c='#d62728', s=4, alpha=0.4, rasterized=True)
        ax.set_title(f'{name}\n{col_titles[3]}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(handles=dataset_legend, loc='upper right', fontsize=8,
                     markerscale=1.0, framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def make_train_test_figure(hiddens_train, hiddens_test,
                           tissue_train, tissue_test,
                           preds_train, preds_test,
                           roi_train, roi_test,
                           output_path):
    """Combined t-SNE: train + test in same embedding.

    3 datasets: Train-Local (###B/S), Train-TCGA (TC_*), Test-TCGA.
    Row = model (lizard / Stage 1 / Stage 2).
    Col 1: 3-way dataset split.
    Col 2: tissue type.
    Col 3: predicted cell class.
    """
    model_names = ['Lizard (baseline)', 'Stage 1 (context_soft)', 'Stage 2 (Loki-OT)']
    n_train = hiddens_train[0].shape[0]
    n_test = hiddens_test[0].shape[0]

    tissue_all = np.concatenate([
        tissue_train.numpy() if torch.is_tensor(tissue_train) else tissue_train,
        tissue_test.numpy() if torch.is_tensor(tissue_test) else tissue_test,
    ])

    # Build 3-way dataset labels: Hospital 1 (train), Hospital 2 (train), TCGA (test)
    train_ds = roi_to_dataset(roi_train)  # 'Hospital 1' or 'Hospital 2'
    dataset_3way = np.concatenate([
        np.array([f'{d} (train)' for d in train_ds]),
        np.array(['TCGA (test)'] * n_test),
    ])

    n_h1 = (dataset_3way == 'Hospital 1 (train)').sum()
    n_h2 = (dataset_3way == 'Hospital 2 (train)').sum()
    n_test_tcga = (dataset_3way == 'TCGA (test)').sum()
    print(f"  3-way split: Hospital 1 (train)={n_h1}, Hospital 2 (train)={n_h2}, TCGA (test)={n_test_tcga}")

    print("  Running joint t-SNE (train+test)...")
    tsne_results = []
    for i, name in enumerate(model_names):
        h_all = np.concatenate([hiddens_train[i], hiddens_test[i]], axis=0)
        perp = min(TSNE_PERP, h_all.shape[0] // 2)
        tsne = TSNE(n_components=2, random_state=TSNE_SEED, perplexity=perp)
        coords = tsne.fit_transform(h_all)
        tsne_results.append(coords)
        print(f"    {name}: done ({h_all.shape[0]} cells)")

    preds_all = [np.concatenate([preds_train[i], preds_test[i]]) for i in range(3)]

    # 3-way dataset colors and legend
    DS_COLORS = {
        'Hospital 1 (train)': '#d62728',   # red
        'Hospital 2 (train)': '#1f77b4',   # blue
        'TCGA (test)':        '#ff7f0e',   # orange
    }
    ds_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DS_COLORS['Hospital 1 (train)'],
               markersize=8, label=f'Hospital 1 - train ({n_h1})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DS_COLORS['Hospital 2 (train)'],
               markersize=8, label=f'Hospital 2 - train ({n_h2})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DS_COLORS['TCGA (test)'],
               markersize=8, label=f'TCGA - test ({n_test_tcga})'),
    ]
    unique_tissues = np.unique(tissue_all)
    tissue_legend = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=TISSUE_COLORS.get(t, '#999999'),
                            markersize=8, label=TISSUE_NAMES.get(t, f'T{t}'))
                     for t in sorted(unique_tissues)]
    class_legend = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=LIZARD_COLORS[c],
                           markersize=8, label=LIZARD_LABELS[c])
                    for c in range(6)]

    fig, axes = plt.subplots(3, 3, figsize=(24, 22))
    fig.suptitle('t-SNE of Post-MLP Hidden Features — 3 Datasets (joint embedding)',
                 fontsize=16, fontweight='bold', y=0.98)
    col_titles = ['Dataset (3-way)', 'Tissue Type', 'Predicted Cell Class']

    for row, (name, coords, pred) in enumerate(zip(model_names, tsne_results, preds_all)):
        # Col 0: 3-way dataset
        ax = axes[row, 0]
        for ds_name in ['TCGA (test)', 'Hospital 2 (train)', 'Hospital 1 (train)']:
            mask = dataset_3way == ds_name
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=DS_COLORS[ds_name], s=4, alpha=0.35, rasterized=True)
        ax.set_title(f'{name}\n{col_titles[0]}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(handles=ds_legend, loc='upper right', fontsize=8,
                     markerscale=1.0, framealpha=0.7)

        # Col 1: tissue type
        ax = axes[row, 1]
        for t in sorted(unique_tissues):
            mask = tissue_all == t
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=TISSUE_COLORS.get(t, '#999999'), s=4, alpha=0.4,
                          rasterized=True)
        ax.set_title(f'{name}\n{col_titles[1]}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(handles=tissue_legend, loc='upper right', fontsize=7,
                     markerscale=0.8, framealpha=0.7)

        # Col 2: predicted class
        ax = axes[row, 2]
        for c in range(6):
            mask = pred == c
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=LIZARD_COLORS[c], s=4, alpha=0.4,
                          rasterized=True)
        ax.set_title(f'{name}\n{col_titles[2]}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(handles=class_legend, loc='upper right', fontsize=7,
                     markerscale=0.8, framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(VIS_DIR, exist_ok=True)

    # Load classifiers
    print("Loading classifiers...")
    liz_clf = load_lizard_classifier()
    stage1_clf = load_context_classifier(STAGE1_CHECKPOINT, 'Stage 1')
    stage2_clf = load_context_classifier(STAGE2_CHECKPOINT, 'Stage 2')

    # --- Figure 1: Train ---
    print("\n" + "=" * 60)
    print("TRAIN DATA")
    print("=" * 60)
    ct_train, ctx_train, tissue_train, roi_train = load_train_data()
    ct_train, ctx_train, tissue_train, roi_train = subsample(
        ct_train, ctx_train, tissue_train, roi_train, MAX_CELLS)

    hiddens_train, preds_train = extract_hidden_features(
        liz_clf, stage1_clf, stage2_clf, ct_train, ctx_train)

    make_figure(hiddens_train, preds_train, tissue_train, roi_train,
                't-SNE of Post-MLP Hidden Features — Train',
                os.path.join(VIS_DIR, 'post_mlp_tsne_train.png'))

    # --- Figure 2: Test (TCGA ROIs) ---
    print("\n" + "=" * 60)
    print("TEST DATA (TCGA ROIs)")
    print("=" * 60)
    ct_test, ctx_test, tissue_test, roi_test = load_test_data()
    ct_test, ctx_test, tissue_test, roi_test = subsample(
        ct_test, ctx_test, tissue_test, roi_test, MAX_CELLS)

    hiddens_test, preds_test = extract_hidden_features(
        liz_clf, stage1_clf, stage2_clf, ct_test, ctx_test)

    make_figure(hiddens_test, preds_test, tissue_test, roi_test,
                't-SNE of Post-MLP Hidden Features — Test (TCGA ROIs)',
                os.path.join(VIS_DIR, 'post_mlp_tsne_test.png'))

    # --- Figure 3: Train vs Test (joint embedding) ---
    print("\n" + "=" * 60)
    print("TRAIN vs TEST (joint t-SNE)")
    print("=" * 60)
    make_train_test_figure(
        hiddens_train, hiddens_test,
        tissue_train, tissue_test,
        preds_train, preds_test,
        roi_train, roi_test,
        os.path.join(VIS_DIR, 'post_mlp_tsne_train_vs_test.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
