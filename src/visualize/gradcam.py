"""
Token-Space Gradient Attribution: Stage 1 vs Stage 2 Classifier Comparison

For a handful of cells where the two classifiers disagree most, computes a
Grad-CAM-style attribution over the ViT token grid (both the raw spatial
heatmap and a channel-weighted variant) plus a cell-vs-context contribution
ratio, so you can see whether Stage 2's prediction change is driven by the
cell token or by its spatial context.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'cellvit_repo'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# CONFIGURATION
# ============================================================

# External data — TIGER-style ROI-level annotations. Point at your own copy.
BASE_DIR = '/DataMount/xl260/wsirois/roi-level-annotations/tissue-cells/'
IMAGE_DIR = os.path.join(BASE_DIR, 'selected_images/')
MAT_DIR = os.path.join(BASE_DIR, 'selected_images_output/mat/')
MASK_DIR = os.path.join(BASE_DIR, 'masks/')

CELLVIT_CHECKPOINT = os.path.join(REPO_ROOT, 'weights', 'sam_h', 'CellViT-SAM-H-x40-AMP.pth')
LIZARD_CHECKPOINT  = os.path.join(REPO_ROOT, 'cellvit_repo', 'checkpoints', 'classifier', 'sam-h', 'lizard.pth')
STAGE1_CHECKPOINT  = os.path.join(REPO_ROOT, 'checkpoints', 'context_soft_v4_epoch018.pth')
STAGE2_CHECKPOINT  = os.path.join(REPO_ROOT, 'checkpoints', 'loki_ot_v15_epoch003.pth')
CACHE_PATH = os.path.join(REPO_ROOT, 'outputs', 'cache', 'cached_cell_tokens.pt')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs', 'gradcam_stage1_vs_stage2')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 1024
TOKEN_PATCH_SIZE = 16
TOKEN_GRID_SIZE = PATCH_SIZE // TOKEN_PATCH_SIZE  # 64
CONTEXT_R = 5
LYMPH_CLASS_IDX = 2

N_CELLS_PER_ROI = 5
N_ROIS = 5

TISSUE_NAMES = {1: 'T1 (tumor)', 2: 'T2 (immune)', 3: 'T3', 4: 'T4 (glands)',
                5: 'T5 (bg)', 6: 'T6 (stroma)', 7: 'T7 (necrosis)'}

CLASS_NAMES = ['Neutrophil', 'Epithelial', 'Lymphocyte', 'Plasma', 'Eosinophil', 'Connective']


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class ContextClassifier(nn.Module):
    def __init__(self, cell_dim=1280, context_dim=1280, hidden_dim=512,
                 num_classes=6, drop_rate=0):
        super().__init__()
        self.cell_dim = cell_dim
        self.context_dim = context_dim
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
# LOADING HELPERS
# ============================================================

def load_classifier(checkpoint_path, lizard_path):
    lizard_cp = torch.load(lizard_path, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')
    hidden_dim = lizard_conf['model'].get('hidden_dim', 100)
    num_classes = lizard_conf['data']['num_classes']
    clf = ContextClassifier(
        cell_dim=1280, context_dim=1280,
        hidden_dim=hidden_dim, num_classes=num_classes, drop_rate=0
    )
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    clf.load_state_dict(cp['model_state_dict'])
    clf.eval().to(DEVICE)
    return clf, hidden_dim, num_classes


def load_cellvit_model():
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
    cp = torch.load(CELLVIT_CHECKPOINT, map_location='cpu')
    model = CellViTSAM(
        model_path=CELLVIT_CHECKPOINT,
        num_nuclei_classes=cp['config']['data.num_nuclei_classes'],
        num_tissue_classes=cp['config']['data.num_tissue_classes'],
        vit_structure="SAM-H", drop_rate=0
    )
    model.load_state_dict(cp['model_state_dict'])
    model.eval().to(DEVICE)
    return model


def preprocess_image(img_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    h, w = img.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:
        img = np.pad(img, ((0, max(0, PATCH_SIZE - h)), (0, max(0, PATCH_SIZE - w)), (0, 0)),
                     mode='constant', constant_values=255)
        h, w = img.shape[:2]
    sh, sw = 0, 0
    if h > PATCH_SIZE or w > PATCH_SIZE:
        sh = max(0, (h - PATCH_SIZE) // 2)
        sw = max(0, (w - PATCH_SIZE) // 2)
        img = img[sh:sh + PATCH_SIZE, sw:sw + PATCH_SIZE]
    img_display = img.copy()
    img_norm = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    return img_t, img_display, sh, sw


def load_inst_map(roi_name, sh, sw):
    mat_path = os.path.join(MAT_DIR, f'{roi_name}.mat')
    inst_map = sio.loadmat(mat_path)['inst_map']
    h, w = inst_map.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:
        inst_map = np.pad(inst_map, ((0, max(0, PATCH_SIZE - h)), (0, max(0, PATCH_SIZE - w))), constant_values=0)
    if inst_map.shape[0] > PATCH_SIZE or inst_map.shape[1] > PATCH_SIZE:
        inst_map = inst_map[sh:sh + PATCH_SIZE, sw:sw + PATCH_SIZE]
    return inst_map


def load_tissue_mask(roi_name, sh, sw):
    mask_path = os.path.join(MASK_DIR, f'{roi_name}.png')
    mask = np.array(Image.open(mask_path))
    h, w = mask.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:
        mask = np.pad(mask, ((0, max(0, PATCH_SIZE - h)), (0, max(0, PATCH_SIZE - w))), constant_values=0)
    if mask.shape[0] > PATCH_SIZE or mask.shape[1] > PATCH_SIZE:
        mask = mask[sh:sh + PATCH_SIZE, sw:sw + PATCH_SIZE]
    return mask


# ============================================================
# GRADIENT ATTRIBUTION
# ============================================================

def compute_cell_bbox(inst_map, cell_id):
    mask = inst_map == cell_id
    if mask.sum() < 10:
        return None
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    ty0 = max(0, int(np.floor(ys.min() / TOKEN_PATCH_SIZE)))
    ty1 = min(TOKEN_GRID_SIZE, int(np.ceil((ys.max() + 1) / TOKEN_PATCH_SIZE)))
    tx0 = max(0, int(np.floor(xs.min() / TOKEN_PATCH_SIZE)))
    tx1 = min(TOKEN_GRID_SIZE, int(np.ceil((xs.max() + 1) / TOKEN_PATCH_SIZE)))
    if ty1 <= ty0 or tx1 <= tx0:
        return None
    ey0 = max(0, ty0 - CONTEXT_R)
    ey1 = min(TOKEN_GRID_SIZE, ty1 + CONTEXT_R)
    ex0 = max(0, tx0 - CONTEXT_R)
    ex1 = min(TOKEN_GRID_SIZE, tx1 + CONTEXT_R)
    return ty0, ty1, tx0, tx1, ey0, ey1, ex0, ex1, cy, cx


def compute_gradcam_for_cell(patch_tokens_np, classifier, bbox_info, target_class=LYMPH_CLASS_IDX):
    ty0, ty1, tx0, tx1, ey0, ey1, ex0, ex1, cy, cx = bbox_info
    pt = torch.tensor(patch_tokens_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    C = pt.shape[0]
    cell_token = pt[:, ty0:ty1, tx0:tx1].reshape(C, -1).mean(dim=1)
    context_token = pt[:, ey0:ey1, ex0:ex1].reshape(C, -1).mean(dim=1)
    logits = classifier(cell_token.unsqueeze(0), context_token.unsqueeze(0))
    logits[0, target_class].backward()
    grad = pt.grad.detach()
    act = pt.detach()
    heatmap = (grad * act).sum(dim=0).relu()
    heatmap_np = heatmap.cpu().numpy()
    channel_imp_cell = grad[:, ty0:ty1, tx0:tx1].mean(dim=(1, 2)).abs()
    channel_heatmap = (channel_imp_cell[:, None, None] * act).sum(dim=0).relu()
    channel_heatmap_np = channel_heatmap.cpu().numpy()
    channel_imp_np = channel_imp_cell.cpu().numpy()
    W = classifier.fc1.weight.detach()
    W_cell = W[:, :1280]
    W_ctx = W[:, 1280:]
    cell_contrib = (W_cell @ cell_token.detach()).norm().item()
    ctx_contrib = (W_ctx @ context_token.detach()).norm().item()
    ratio = ctx_contrib / (cell_contrib + 1e-8)
    logits_np = logits.detach().cpu().numpy()[0]
    return heatmap_np, channel_imp_np, channel_heatmap_np, logits_np, ratio


def normalize_heatmap(hm):
    mn, mx = hm.min(), hm.max()
    if mx - mn < 1e-10:
        return np.zeros_like(hm)
    return (hm - mn) / (mx - mn)


# ============================================================
# VISUALIZATION
# ============================================================

def make_cell_figure(img_display, bbox_info, hm_s1, hm_s2, ch_hm_s1, ch_hm_s2,
                     ch_imp_s1, ch_imp_s2, logits_s1, logits_s2,
                     ratio_s1, ratio_s2, roi_name, cell_id, tissue_type):
    ty0, ty1, tx0, tx1, ey0, ey1, ex0, ex1, cy, cx = bbox_info

    cell_px = (tx0 * TOKEN_PATCH_SIZE, ty0 * TOKEN_PATCH_SIZE,
               (tx1 - tx0) * TOKEN_PATCH_SIZE, (ty1 - ty0) * TOKEN_PATCH_SIZE)
    ctx_px = (ex0 * TOKEN_PATCH_SIZE, ey0 * TOKEN_PATCH_SIZE,
              (ex1 - ex0) * TOKEN_PATCH_SIZE, (ey1 - ey0) * TOKEN_PATCH_SIZE)

    hm_s1_n = normalize_heatmap(hm_s1)
    hm_s2_n = normalize_heatmap(hm_s2)
    ch_hm_s1_n = normalize_heatmap(ch_hm_s1)
    ch_hm_s2_n = normalize_heatmap(ch_hm_s2)

    def upsample(hm):
        t = torch.tensor(hm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        up = F.interpolate(t, size=(PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=False)
        return up[0, 0].numpy()

    hm_s1_up = upsample(hm_s1_n)
    hm_s2_up = upsample(hm_s2_n)
    ch_hm_s1_up = upsample(ch_hm_s1_n)
    ch_hm_s2_up = upsample(ch_hm_s2_n)

    diff_hm = normalize_heatmap(ch_hm_s2_n - ch_hm_s1_n)
    diff_hm_up = upsample(diff_hm)

    img_f = img_display.astype(np.float32) / 255.0

    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    t_name = TISSUE_NAMES.get(tissue_type, f'T{tissue_type}')
    s1_prob = F.softmax(torch.tensor(logits_s1), dim=0)[LYMPH_CLASS_IDX].item()
    s2_prob = F.softmax(torch.tensor(logits_s2), dim=0)[LYMPH_CLASS_IDX].item()

    fig.suptitle(
        f'{roi_name}  |  cell {cell_id}  |  {t_name}\n'
        f'Lymph prob: Stage1={s1_prob:.3f}  Stage2={s2_prob:.3f}  '
        f'({"suppressed" if s2_prob < s1_prob - 0.05 else "similar"})',
        fontsize=14, fontweight='bold'
    )

    # --- Row 1: Spatial focus (Grad-CAM) ---
    ax = axes[0, 0]
    ax.imshow(img_f)
    ax.add_patch(patches.Rectangle((cell_px[0], cell_px[1]), cell_px[2], cell_px[3],
                                    linewidth=2, edgecolor='lime', facecolor='none'))
    ax.add_patch(patches.Rectangle((ctx_px[0], ctx_px[1]), ctx_px[2], ctx_px[3],
                                    linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'))
    ax.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
    ax.set_title('Original + bboxes\n(green=cell, cyan=context)', fontsize=10)
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(img_f)
    ax.imshow(hm_s1_up, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.set_title(f'Stage 1 Spatial Grad-CAM\nctx/cell ratio={ratio_s1:.3f}', fontsize=10)
    ax.axis('off')

    ax = axes[0, 2]
    ax.imshow(img_f)
    ax.imshow(hm_s2_up, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.set_title(f'Stage 2 Spatial Grad-CAM\nctx/cell ratio={ratio_s2:.3f}', fontsize=10)
    ax.axis('off')

    # --- Row 2: Zoomed-in Grad-CAM (context window only) ---
    cy0_px = ey0 * TOKEN_PATCH_SIZE
    cy1_px = ey1 * TOKEN_PATCH_SIZE
    cx0_px = ex0 * TOKEN_PATCH_SIZE
    cx1_px = ex1 * TOKEN_PATCH_SIZE
    cell_rel_x = cell_px[0] - cx0_px
    cell_rel_y = cell_px[1] - cy0_px

    ax = axes[1, 0]
    ax.imshow(img_f[cy0_px:cy1_px, cx0_px:cx1_px])
    ax.add_patch(patches.Rectangle((cell_rel_x, cell_rel_y), cell_px[2], cell_px[3],
                                    linewidth=2, edgecolor='lime', facecolor='none'))
    ax.plot(cx - cx0_px, cy - cy0_px, 'r+', markersize=12, markeredgewidth=2)
    ax.set_title('Zoomed: Original\n(context window)', fontsize=10)
    ax.axis('off')

    ax = axes[1, 1]
    ax.imshow(img_f[cy0_px:cy1_px, cx0_px:cx1_px])
    ax.imshow(hm_s1_up[cy0_px:cy1_px, cx0_px:cx1_px], cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.add_patch(patches.Rectangle((cell_rel_x, cell_rel_y), cell_px[2], cell_px[3],
                                    linewidth=2, edgecolor='lime', facecolor='none'))
    ax.set_title('Zoomed: Stage 1 Grad-CAM', fontsize=10)
    ax.axis('off')

    ax = axes[1, 2]
    ax.imshow(img_f[cy0_px:cy1_px, cx0_px:cx1_px])
    ax.imshow(hm_s2_up[cy0_px:cy1_px, cx0_px:cx1_px], cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.add_patch(patches.Rectangle((cell_rel_x, cell_rel_y), cell_px[2], cell_px[3],
                                    linewidth=2, edgecolor='lime', facecolor='none'))
    ax.set_title('Zoomed: Stage 2 Grad-CAM', fontsize=10)
    ax.axis('off')

    # --- Row 3: Channel-weighted heatmaps ---
    ax = axes[2, 0]
    ax.imshow(img_f)
    ax.imshow(ch_hm_s1_up, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    ax.set_title('Stage 1 Channel-Weighted', fontsize=10)
    ax.axis('off')

    ax = axes[2, 1]
    ax.imshow(img_f)
    ax.imshow(ch_hm_s2_up, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    ax.set_title('Stage 2 Channel-Weighted', fontsize=10)
    ax.axis('off')

    ax = axes[2, 2]
    ax.imshow(img_f)
    ax.imshow(diff_hm_up, cmap='coolwarm', alpha=0.5, vmin=0, vmax=1)
    ax.set_title('Differential (Stage 2 - Stage 1)', fontsize=10)
    ax.axis('off')

    # --- Row 4: Quantitative ---
    ax = axes[3, 0]
    x_pos = np.arange(2)
    width = 0.35
    ax.bar(x_pos - width/2, [1.0, 1.0], width, label='Cell', color='#2196F3')
    ax.bar(x_pos + width/2, [ratio_s1, ratio_s2], width, label='Context', color='#FF9800')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Stage 1', 'Stage 2'])
    ax.set_ylabel('Relative contribution')
    ax.set_title('Cell vs Context Weight', fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[3, 1]
    top_k = 20
    top_s1_idx = np.argsort(ch_imp_s1)[-top_k:][::-1]
    top_s2_idx = np.argsort(ch_imp_s2)[-top_k:][::-1]
    overlap = set(top_s1_idx.tolist()) & set(top_s2_idx.tolist())
    ax.bar(range(top_k), ch_imp_s1[top_s1_idx], alpha=0.6, label='Stage 1 top-20', color='#2196F3')
    ax.bar(range(top_k), ch_imp_s2[top_s2_idx], alpha=0.6, label='Stage 2 top-20', color='#F44336')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Channel importance')
    ax.set_title(f'Top-20 Channels\noverlap={len(overlap)}/20', fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[3, 2]
    ax.axis('off')
    stats_text = (
        f"Cell ID: {cell_id}\n"
        f"Tissue: {t_name}\n"
        f"─────────────────\n"
        f"Stage1 lymph prob:  {s1_prob:.4f}\n"
        f"Stage2 lymph prob:  {s2_prob:.4f}\n"
        f"Change:             {s2_prob - s1_prob:+.4f}\n"
        f"─────────────────\n"
        f"Stage1 ctx/cell:    {ratio_s1:.4f}\n"
        f"Stage2 ctx/cell:    {ratio_s2:.4f}\n"
        f"Ratio change:       {ratio_s2 - ratio_s1:+.4f}\n"
        f"─────────────────\n"
        f"Top-20 ch overlap: {len(overlap)}/20\n"
        f"Stage1 pred class:  {CLASS_NAMES[logits_s1.argmax()]}\n"
        f"Stage2 pred class:  {CLASS_NAMES[logits_s2.argmax()]}\n"
    )
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


# ============================================================
# AUTO-SELECT INTERESTING CELLS
# ============================================================

def auto_select_rois_and_cells(cache, clf_stage1, clf_stage2, n_rois=N_ROIS, n_cells=N_CELLS_PER_ROI):
    print(f"Auto-selecting interesting cells (max Stage1-Stage2 disagreement)...")
    roi_scores = {}
    roi_cell_scores = {}
    for roi_name, roi in cache.items():
        cell_tokens = roi['cell_tokens'].float().to(DEVICE)
        context_tokens = roi['context_tokens'].float().to(DEVICE)
        with torch.no_grad():
            logits_s1 = clf_stage1(cell_tokens, context_tokens)
            logits_s2 = clf_stage2(cell_tokens, context_tokens)
            probs_s1 = F.softmax(logits_s1, dim=1)[:, LYMPH_CLASS_IDX]
            probs_s2 = F.softmax(logits_s2, dim=1)[:, LYMPH_CLASS_IDX]
            diff = (probs_s1 - probs_s2).abs().cpu().numpy()
        roi_scores[roi_name] = diff.mean()
        roi_cell_scores[roi_name] = diff
    top_rois = sorted(roi_scores, key=roi_scores.get, reverse=True)[:n_rois]
    selections = {}
    for roi_name in top_rois:
        diff = roi_cell_scores[roi_name]
        top_cell_indices = np.argsort(diff)[-n_cells:][::-1]
        selections[roi_name] = top_cell_indices.tolist()
        print(f"  {roi_name}: {len(top_cell_indices)} cells, "
              f"max_diff={diff[top_cell_indices[0]]:.3f}")
    return selections


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Token-Space Gradient Attribution: Stage 1 vs Stage 2")
    print("=" * 70)

    print("\nLoading classifiers...")
    clf_stage1, hidden_dim, num_classes = load_classifier(STAGE1_CHECKPOINT, LIZARD_CHECKPOINT)
    clf_stage2, _, _ = load_classifier(STAGE2_CHECKPOINT, LIZARD_CHECKPOINT)
    print(f"  Architecture: [1280+1280] -> {hidden_dim} -> {num_classes}")

    print("\nLoading cache for cell selection...")
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    print(f"  {len(cache)} ROIs loaded")

    selections = auto_select_rois_and_cells(cache, clf_stage1, clf_stage2)

    print("\nLoading CellViT-SAM-H model...")
    cellvit_model = load_cellvit_model()
    print("  Model loaded")

    n_generated = 0
    for roi_name, cell_indices in selections.items():
        print(f"\n{'─' * 60}")
        print(f"Processing ROI: {roi_name}")
        print(f"{'─' * 60}")

        img_path = os.path.join(IMAGE_DIR, f'{roi_name}.png')
        if not os.path.exists(img_path):
            print(f"  Image not found, skipping")
            continue

        img_t, img_display, sh, sw = preprocess_image(img_path)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = cellvit_model(img_t, retrieve_tokens=True)
        patch_tokens_np = preds['tokens'][0].cpu().float().numpy()

        inst_map = load_inst_map(roi_name, sh, sw)
        tissue_mask = load_tissue_mask(roi_name, sh, sw)

        cell_ids = [cid for cid in np.unique(inst_map) if cid != 0]
        valid_cell_ids = []
        for cid in cell_ids:
            mask = inst_map == cid
            if mask.sum() >= 10:
                valid_cell_ids.append(cid)

        for cell_idx in cell_indices:
            if cell_idx >= len(valid_cell_ids):
                continue
            cell_id = valid_cell_ids[cell_idx]

            bbox_info = compute_cell_bbox(inst_map, cell_id)
            if bbox_info is None:
                continue

            cy, cx = bbox_info[8], bbox_info[9]
            tissue_type = int(tissue_mask[min(int(cy), PATCH_SIZE - 1),
                                          min(int(cx), PATCH_SIZE - 1)])

            print(f"  Cell {cell_id} (idx={cell_idx}, tissue={TISSUE_NAMES.get(tissue_type, f'T{tissue_type}')})")

            hm_s1, ch_imp_s1, ch_hm_s1, logits_s1, ratio_s1 = compute_gradcam_for_cell(
                patch_tokens_np, clf_stage1, bbox_info, LYMPH_CLASS_IDX
            )
            hm_s2, ch_imp_s2, ch_hm_s2, logits_s2, ratio_s2 = compute_gradcam_for_cell(
                patch_tokens_np, clf_stage2, bbox_info, LYMPH_CLASS_IDX
            )

            s1_lp = F.softmax(torch.tensor(logits_s1), dim=0)[LYMPH_CLASS_IDX].item()
            s2_lp = F.softmax(torch.tensor(logits_s2), dim=0)[LYMPH_CLASS_IDX].item()
            print(f"    Stage1 lymph={s1_lp:.3f}, Stage2 lymph={s2_lp:.3f}, "
                  f"ctx_ratio Stage1={ratio_s1:.3f} Stage2={ratio_s2:.3f}")

            fig = make_cell_figure(
                img_display, bbox_info,
                hm_s1, hm_s2, ch_hm_s1, ch_hm_s2,
                ch_imp_s1, ch_imp_s2, logits_s1, logits_s2,
                ratio_s1, ratio_s2, roi_name, cell_id, tissue_type
            )

            safe_name = roi_name.replace('[', '_').replace(']', '_').replace(',', '').replace(' ', '')
            fig_path = os.path.join(OUTPUT_DIR, f'{safe_name}_cell{cell_id}.png')
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path}")
            n_generated += 1

    del cellvit_model
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"Done. Generated {n_generated} figures in {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
