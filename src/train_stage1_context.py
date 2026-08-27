"""
Loki-OT Stage 1: Context-Aware Classifier Fine-Tuning

Each cell gets a CONTEXT TOKEN: average of ViT patch tokens in a wider
spatial neighborhood (cell bbox expanded by CONTEXT_R tokens in each direction),
concatenated with its own cell token to form the ContextClassifier input.

Weight initialization: lizard weights for cell columns, zeros for context columns
-> epoch 0 output is identical to the lizard baseline; the model learns to use
context during training via an MLLM-guided suppress/distill loss.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'cellvit_repo'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import scipy.io as sio
from PIL import Image


class LinearClassifier(nn.Module):
    """Original lizard classifier (for weight extraction during init)."""
    def __init__(self, embed_dim=1280, hidden_dim=100, num_classes=6, drop_rate=0):
        super(LinearClassifier, self).__init__()
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


def unflatten_dict(d, sep='.'):
    """Unflatten a flattened dictionary into a nested one."""
    output_dict = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = output_dict
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    return output_dict


# ============================================================
# CONFIGURATION
# ============================================================

# External data — TIGER-style ROI-level annotations (images/.mat instance maps/
# tissue masks). Point this at your own copy; not shipped in this repo.
# Set LOKI_OT_DATA_ROOT to wherever you've placed the TIGER WSIROIS data (see README "Dataset").
DATA_ROOT = os.environ.get('LOKI_OT_DATA_ROOT', '/DataMount/xl260/wsirois')
BASE_DIR = os.path.join(DATA_ROOT, 'roi-level-annotations', 'tissue-cells') + '/'
IMAGE_DIR = os.path.join(BASE_DIR, 'selected_images/')
MAT_DIR = os.path.join(BASE_DIR, 'selected_images_output/mat/')
MASK_DIR = os.path.join(BASE_DIR, 'masks/')

CELLVIT_CHECKPOINT = os.path.join(REPO_ROOT, 'weights', 'sam_h', 'CellViT-SAM-H-x40-AMP.pth')
LIZARD_CHECKPOINT = os.path.join(REPO_ROOT, 'cellvit_repo', 'checkpoints', 'classifier', 'sam-h', 'lizard.pth')
MLLM_JSON = os.path.join(REPO_ROOT, 'data', 'mllm_reference.json')
CACHE_PATH = os.path.join(REPO_ROOT, 'outputs', 'cache', 'cached_cell_tokens.pt')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs', 'stage1_context')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 1024
TOKEN_PATCH_SIZE = 16
TOKEN_GRID_SIZE = PATCH_SIZE // TOKEN_PATCH_SIZE  # 64

# Training hyperparameters
LR = 1e-4
N_EPOCHS = 20
LYMPH_CLASS_IDX = 2         # lizard: Lymphocyte

# MLLM-guided loss parameters
DIVERGE_THRESHOLD = 0.10
TAU = 0.05
SUPPRESS_SCALE = 2.0
DISTILL_BOOST = 3.0

# Context radius
CONTEXT_R = 5  # expand cell bbox by 5 tokens (80 pixels) in each direction

TISSUE_NAMES = {1: 'T1 (tumor)', 2: 'T2 (immune)', 3: 'T3', 4: 'T4 (glands)',
                5: 'T5 (bg)', 6: 'T6 (stroma)', 7: 'T7 (necrosis)'}


# ============================================================
# CONTEXT CLASSIFIER
# ============================================================

class ContextClassifier(nn.Module):
    """Context-aware cell classifier.

    Concatenates cell token (cell_dim) with spatial context token (context_dim)
    to form input. Architecture mirrors LinearClassifier but with doubled input.

    At initialization with lizard weights + zero context columns, output is
    identical to lizard baseline. Model learns to use context during training.
    """

    def __init__(self, cell_dim=1280, context_dim=1280, hidden_dim=512,
                 num_classes=6, drop_rate=0):
        super(ContextClassifier, self).__init__()
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


# ============================================================
# WEIGHT INITIALIZATION
# ============================================================

def init_context_classifier_from_lizard(lizard_checkpoint_path):
    """Initialize ContextClassifier from lizard weights.

    fc1.weight (hidden_dim, 2560): lizard fc1 in first 1280 cols, zeros in last 1280
    fc1.bias, fc2.weight, fc2.bias: copied directly from lizard

    Returns:
        classifier: ContextClassifier with initialized weights
        lizard_cp: raw checkpoint dict (needed for config preservation)
        hidden_dim: int
        num_classes: int
    """
    lizard_cp = torch.load(lizard_checkpoint_path, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')

    hidden_dim = lizard_conf['model'].get('hidden_dim', 100)
    num_classes = lizard_conf['data']['num_classes']

    # Load lizard LinearClassifier to extract weights
    lizard_clf = LinearClassifier(
        embed_dim=1280, hidden_dim=hidden_dim,
        num_classes=num_classes, drop_rate=0
    )
    lizard_clf.load_state_dict(lizard_cp['model_state_dict'])

    # Build ContextClassifier
    classifier = ContextClassifier(
        cell_dim=1280, context_dim=1280,
        hidden_dim=hidden_dim, num_classes=num_classes, drop_rate=0
    )

    # Copy weights: cell columns from lizard, context columns zeroed
    with torch.no_grad():
        classifier.fc1.weight.zero_()
        classifier.fc1.weight[:, :1280].copy_(lizard_clf.fc1.weight)
        classifier.fc1.bias.copy_(lizard_clf.fc1.bias)
        classifier.fc2.weight.copy_(lizard_clf.fc2.weight)
        classifier.fc2.bias.copy_(lizard_clf.fc2.bias)

    return classifier, lizard_cp, hidden_dim, num_classes


# ============================================================
# CACHE EXTRACTION (cell tokens + context tokens)
# ============================================================

def extract_and_cache():
    """Extract cell tokens AND context tokens for all ROIs.

    For each cell:
      - cell_token: average of patch tokens in cell's bbox
      - context_token: average of patch tokens in expanded bbox
        (cell bbox + CONTEXT_R tokens in each direction)

    Cache structure per ROI:
        'cell_tokens':    (N, 1280)
        'context_tokens': (N, 1280)
        'cell_tissue':    (N,)
        'lizard_soft':    (N, 6)
    """
    if os.path.exists(CACHE_PATH):
        print(f"Cache exists: {CACHE_PATH}")
        return torch.load(CACHE_PATH, map_location='cpu', weights_only=False)

    print("=" * 60)
    print("Phase 1: Extracting CellViT tokens + context for all ROIs")
    print(f"  CONTEXT_R = {CONTEXT_R} tokens ({CONTEXT_R * TOKEN_PATCH_SIZE} pixels)")
    print("=" * 60)

    # Deferred import: CellViTSAM only needed for cache extraction
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM

    # Load CellViT model
    print("Loading CellViT SAM-H model...")
    cp = torch.load(CELLVIT_CHECKPOINT, map_location='cpu')
    model = CellViTSAM(
        model_path=CELLVIT_CHECKPOINT,
        num_nuclei_classes=cp['config']['data.num_nuclei_classes'],
        num_tissue_classes=cp['config']['data.num_tissue_classes'],
        vit_structure="SAM-H", drop_rate=0
    )
    model.load_state_dict(cp['model_state_dict'])
    model.eval().to(DEVICE)

    # Load lizard classifier (for teacher softmax)
    print("Loading lizard classifier (teacher)...")
    lizard_cp = torch.load(LIZARD_CHECKPOINT, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')
    lizard_clf = LinearClassifier(
        embed_dim=1280, hidden_dim=lizard_conf['model'].get('hidden_dim', 100),
        num_classes=lizard_conf['data']['num_classes'], drop_rate=0
    )
    lizard_clf.load_state_dict(lizard_cp['model_state_dict'])
    lizard_clf.eval().to(DEVICE)

    # Process all ROIs
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])
    print(f"Processing {len(image_files)} ROIs...")

    cache = {}
    for idx, img_file in enumerate(image_files):
        roi_name = img_file.replace('.png', '')

        mat_path = os.path.join(MAT_DIR, f'{roi_name}.mat')
        mask_path = os.path.join(MASK_DIR, img_file)
        if not os.path.exists(mat_path) or not os.path.exists(mask_path):
            continue

        # Load and preprocess image
        img = np.array(Image.open(os.path.join(IMAGE_DIR, img_file)).convert('RGB'))
        h, w = img.shape[:2]
        if h < PATCH_SIZE or w < PATCH_SIZE:
            img = np.pad(img, ((0, max(0, PATCH_SIZE-h)), (0, max(0, PATCH_SIZE-w)), (0, 0)),
                         mode='constant', constant_values=255)
            h, w = img.shape[:2]
        sh, sw = 0, 0
        if h > PATCH_SIZE or w > PATCH_SIZE:
            sh = max(0, (h - PATCH_SIZE) // 2)
            sw = max(0, (w - PATCH_SIZE) // 2)
            img = img[sh:sh+PATCH_SIZE, sw:sw+PATCH_SIZE]

        img_norm = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        # CellViT forward (with tokens)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(img_t, retrieve_tokens=True)
        patch_tokens = preds['tokens'].cpu()[0]  # (1280, 64, 64)
        C = patch_tokens.shape[0]

        # Load instance map + tissue mask (with same crop)
        inst_map = sio.loadmat(mat_path)['inst_map']
        tissue_mask = np.array(Image.open(mask_path))
        if inst_map.shape[0] > PATCH_SIZE or inst_map.shape[1] > PATCH_SIZE:
            inst_map = inst_map[sh:sh+PATCH_SIZE, sw:sw+PATCH_SIZE]
            tissue_mask = tissue_mask[sh:sh+PATCH_SIZE, sw:sw+PATCH_SIZE]
        elif inst_map.shape[0] < PATCH_SIZE or inst_map.shape[1] < PATCH_SIZE:
            ph = max(0, PATCH_SIZE - inst_map.shape[0])
            pw = max(0, PATCH_SIZE - inst_map.shape[1])
            inst_map = np.pad(inst_map, ((0, ph), (0, pw)), constant_values=0)
            tissue_mask = np.pad(tissue_mask, ((0, ph), (0, pw)), constant_values=0)

        # Extract per-cell tokens + context tokens + tissue type
        cell_tokens_list = []
        context_tokens_list = []
        cell_tissue_list = []

        for cid in np.unique(inst_map):
            if cid == 0:
                continue
            mask = inst_map == cid
            if mask.sum() < 10:
                continue
            ys, xs = np.where(mask)
            cy, cx = ys.mean(), xs.mean()

            # Cell token bbox
            ty0 = max(0, int(np.floor(ys.min() / TOKEN_PATCH_SIZE)))
            ty1 = min(TOKEN_GRID_SIZE, int(np.ceil((ys.max() + 1) / TOKEN_PATCH_SIZE)))
            tx0 = max(0, int(np.floor(xs.min() / TOKEN_PATCH_SIZE)))
            tx1 = min(TOKEN_GRID_SIZE, int(np.ceil((xs.max() + 1) / TOKEN_PATCH_SIZE)))
            if ty1 <= ty0 or tx1 <= tx0:
                continue

            # Cell token
            ct = patch_tokens[:, ty0:ty1, tx0:tx1].reshape(C, -1).mean(dim=1)

            # Context token: expanded bbox
            ey0 = max(0, ty0 - CONTEXT_R)
            ey1 = min(TOKEN_GRID_SIZE, ty1 + CONTEXT_R)
            ex0 = max(0, tx0 - CONTEXT_R)
            ex1 = min(TOKEN_GRID_SIZE, tx1 + CONTEXT_R)
            ctx = patch_tokens[:, ey0:ey1, ex0:ex1].reshape(C, -1).mean(dim=1)

            tt = tissue_mask[min(int(cy), PATCH_SIZE-1), min(int(cx), PATCH_SIZE-1)]

            cell_tokens_list.append(ct)
            context_tokens_list.append(ctx)
            cell_tissue_list.append(int(tt))

        if len(cell_tokens_list) == 0:
            continue

        cell_tokens = torch.stack(cell_tokens_list)       # (N, 1280)
        context_tokens = torch.stack(context_tokens_list)  # (N, 1280)
        cell_tissue = torch.tensor(cell_tissue_list)       # (N,)

        # Lizard teacher softmax
        with torch.no_grad():
            logits = lizard_clf(cell_tokens.float().to(DEVICE))
            lizard_soft = F.softmax(logits, dim=1).cpu()   # (N, 6)

        cache[roi_name] = {
            'cell_tokens': cell_tokens,
            'context_tokens': context_tokens,
            'cell_tissue': cell_tissue,
            'lizard_soft': lizard_soft,
        }

        if (idx + 1) % 20 == 0:
            total_cells = sum(v['cell_tokens'].shape[0] for v in cache.values())
            print(f"  {idx+1}/{len(image_files)} ROIs, {total_cells} cells total")

    # Save cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    torch.save(cache, CACHE_PATH)
    total_cells = sum(v['cell_tokens'].shape[0] for v in cache.values())
    print(f"\nCached {len(cache)} ROIs, {total_cells} total cells -> {CACHE_PATH}")

    # Free GPU memory
    del model, lizard_clf
    torch.cuda.empty_cache()

    return cache


# ============================================================
# LOAD CACHE + MLLM
# ============================================================

def attach_mllm_estimates(cache):
    """Attach MLLM lymphocyte % estimates to each ROI in cache."""
    print("Loading MLLM reference...")
    with open(MLLM_JSON, 'r') as f:
        mllm_all = json.load(f)

    n_with_mllm = 0
    for roi_name in cache:
        key = roi_name + '.png'
        mllm_est = mllm_all.get(key, {})
        cache[roi_name]['mllm_estimates'] = mllm_est
        if len(mllm_est) > 0:
            n_with_mllm += 1

    print(f"  {n_with_mllm}/{len(cache)} ROIs have MLLM estimates")
    return cache


# ============================================================
# LOSS FUNCTION
# ============================================================

def mllm_guided_loss(new_soft, cell_tissue, lizard_soft, mllm_estimates):
    """
    MLLM-guided dynamic suppress/distill loss with cell-count weighting.
    Operates on softmax output, agnostic to input representation.
    """
    device = new_soft.device
    total_weighted_loss = torch.tensor(0.0, device=device)
    total_cells_in_roi = 0
    info = {'tissues': []}

    tissue_types = cell_tissue.unique().tolist()
    tissue_types = [t for t in tissue_types if t != 0]

    for tissue_type in tissue_types:
        tissue_mask = (cell_tissue == tissue_type)
        n_cells = tissue_mask.sum().item()
        if n_cells == 0:
            continue

        new_tissue = new_soft[tissue_mask]
        lizard_tissue = lizard_soft[tissue_mask]

        lymph_probs = new_tissue[:, LYMPH_CLASS_IDX]
        pred_pct = lymph_probs.mean()

        tissue_str = str(tissue_type)
        has_mllm = tissue_str in mllm_estimates

        if has_mllm:
            mllm_target = mllm_estimates[tissue_str] / 100.0
            deviation = pred_pct - mllm_target

            if deviation.item() > DIVERGE_THRESHOLD:
                over_pred = torch.relu(deviation - TAU)
                base_loss = over_pred.pow(2)
                tissue_loss = SUPPRESS_SCALE * base_loss
                loss_type = 'suppress'
            else:
                new_log = torch.log(new_tissue.clamp(min=1e-7))
                lizard_log = lizard_tissue.clamp(min=1e-7)
                per_cell_kl = F.kl_div(new_log, lizard_log, reduction='none').sum(dim=1)
                base_loss = per_cell_kl.mean()
                tissue_loss = DISTILL_BOOST * base_loss
                loss_type = 'distill'
        else:
            mllm_target = None
            deviation = None
            new_log = torch.log(new_tissue.clamp(min=1e-7))
            lizard_log = lizard_tissue.clamp(min=1e-7)
            per_cell_kl = F.kl_div(new_log, lizard_log, reduction='none').sum(dim=1)
            base_loss = per_cell_kl.mean()
            tissue_loss = 1.0 * base_loss
            loss_type = 'distill_default'

        total_weighted_loss = total_weighted_loss + tissue_loss * n_cells
        total_cells_in_roi += n_cells

        info['tissues'].append({
            'tissue_type': tissue_type,
            'n_cells': n_cells,
            'pred_pct': pred_pct.item(),
            'mllm_target': mllm_target if mllm_target is None else mllm_target,
            'deviation': deviation.item() if deviation is not None else None,
            'loss_type': loss_type,
            'base_loss': base_loss.item(),
            'tissue_loss': tissue_loss.item(),
        })

    n_groups = len(info['tissues'])
    if total_cells_in_roi > 0:
        final_loss = total_weighted_loss / total_cells_in_roi
    else:
        final_loss = total_weighted_loss

    info['n_groups'] = n_groups
    info['total_loss'] = final_loss.item() if n_groups > 0 else 0.0

    return final_loss, info


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(classifier, cache, optimizer, epoch):
    """One training epoch over all cached ROIs."""
    classifier.train()

    epoch_losses = []
    tissue_details = []

    roi_names = list(cache.keys())
    roi_order = np.random.permutation(len(roi_names))

    for count, idx in enumerate(roi_order):
        roi_name = roi_names[idx]
        roi = cache[roi_name]

        cell_tokens = roi['cell_tokens'].float().to(DEVICE)
        context_tokens = roi['context_tokens'].float().to(DEVICE)
        cell_tissue = roi['cell_tissue'].to(DEVICE)
        lizard_soft = roi['lizard_soft'].to(DEVICE)
        mllm_estimates = roi.get('mllm_estimates', {})

        optimizer.zero_grad()

        # Forward: ContextClassifier on cell + context tokens
        logits = classifier(cell_tokens, context_tokens)
        new_soft = F.softmax(logits, dim=1)

        # Loss
        loss, info = mllm_guided_loss(new_soft, cell_tissue, lizard_soft, mllm_estimates)

        if info['n_groups'] > 0:
            loss.backward()
            optimizer.step()
            epoch_losses.append(info['total_loss'])
            tissue_details.extend(info['tissues'])

    # Epoch summary
    avg_loss = np.mean(epoch_losses) if epoch_losses else 0

    tissue_summary = {}
    for t_info in tissue_details:
        t = t_info['tissue_type']
        if t not in tissue_summary:
            tissue_summary[t] = {
                'pred': [], 'loss_values': [], 'n_cells': [],
                'loss_types': [], 'deviations': []
            }
        tissue_summary[t]['pred'].append(t_info['pred_pct'])
        tissue_summary[t]['loss_values'].append(t_info['base_loss'])
        tissue_summary[t]['n_cells'].append(t_info['n_cells'])
        tissue_summary[t]['loss_types'].append(t_info['loss_type'])
        if t_info['deviation'] is not None:
            tissue_summary[t]['deviations'].append(t_info['deviation'])

    print(f"  Avg loss: {avg_loss:.6f}")
    for t in sorted(tissue_summary.keys()):
        s = tissue_summary[t]
        t_name = TISSUE_NAMES.get(t, f'T{t}')
        n_suppress = sum(1 for lt in s['loss_types'] if lt == 'suppress')
        n_distill = sum(1 for lt in s['loss_types'] if lt.startswith('distill'))
        n_total = len(s['loss_types'])
        avg_dev = np.mean(s['deviations']) if s['deviations'] else 0
        total_cells_epoch = sum(sum(tissue_summary[tt]['n_cells']) for tt in tissue_summary)
        tissue_cells = sum(s['n_cells'])
        eff_weight = tissue_cells / total_cells_epoch * 100 if total_cells_epoch > 0 else 0
        print(f"  {t_name}: cells={tissue_cells}, "
              f"lymph%={np.mean(s['pred']):.3f}, "
              f"avg_dev={avg_dev:+.3f}, "
              f"suppress={n_suppress}/{n_total}, distill={n_distill}/{n_total}, "
              f"weight={eff_weight:.1f}%")

    return avg_loss, tissue_summary


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase 1: Extract cache (cell tokens + context tokens)
    cache = extract_and_cache()
    total_cells = sum(v['cell_tokens'].shape[0] for v in cache.values())
    print(f"\nTotal: {len(cache)} ROIs, {total_cells} cells")

    # Attach MLLM estimates
    cache = attach_mllm_estimates(cache)

    # Count tissue distribution
    tissue_counts = {}
    for v in cache.values():
        for t in v['cell_tissue'].unique().tolist():
            if t == 0:
                continue
            tissue_counts[t] = tissue_counts.get(t, 0) + (v['cell_tissue'] == t).sum().item()
    for t in sorted(tissue_counts):
        t_name = TISSUE_NAMES.get(t, f'T{t}')
        print(f"  {t_name}: {tissue_counts[t]} cells")

    # Initialize ContextClassifier from lizard weights
    print("\nInitializing ContextClassifier from lizard.pth...")
    classifier, lizard_cp, hidden_dim, num_classes = init_context_classifier_from_lizard(LIZARD_CHECKPOINT)
    classifier.to(DEVICE)

    trainable = sum(p.numel() for p in classifier.parameters())
    print(f"  Trainable params: {trainable:,}")
    print(f"  Architecture: [{1280}+{1280}] -> {hidden_dim} -> {num_classes}")

    # Verify weight initialization
    print("  Verifying weight initialization (should match lizard exactly)...")
    classifier.eval()
    sample_roi = list(cache.values())[0]
    with torch.no_grad():
        sample_cell = sample_roi['cell_tokens'][:5].float().to(DEVICE)
        sample_ctx = sample_roi['context_tokens'][:5].float().to(DEVICE)
        out = classifier(sample_cell, sample_ctx)

        lizard_clf_temp = LinearClassifier(
            embed_dim=1280, hidden_dim=hidden_dim,
            num_classes=num_classes, drop_rate=0
        )
        lizard_clf_temp.load_state_dict(lizard_cp['model_state_dict'])
        lizard_clf_temp.eval().to(DEVICE)
        lizard_out = lizard_clf_temp(sample_cell)
        del lizard_clf_temp

        max_diff = (out - lizard_out).abs().max().item()
        print(f"  Max output difference: {max_diff:.2e} (should be ~0)")
        assert max_diff < 1e-3, f"Weight init verification failed: max_diff={max_diff}"

    # Print hyperparameters
    print(f"\nHyperparameters:")
    print(f"  CONTEXT_R = {CONTEXT_R} tokens ({CONTEXT_R * TOKEN_PATCH_SIZE} pixels)")
    print(f"  DIVERGE_THRESHOLD = {DIVERGE_THRESHOLD}")
    print(f"  TAU = {TAU}")
    print(f"  SUPPRESS_SCALE = {SUPPRESS_SCALE}")
    print(f"  DISTILL_BOOST = {DISTILL_BOOST}")
    print(f"  LR = {LR}")
    print(f"  Loss averaging: cell-count weighted")

    # Optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    # Training
    print(f"\n{'=' * 60}")
    print("Stage 1 Training: Context-Aware + Cell-Count Weighted + Threshold 0.10")
    print(f"{'=' * 60}")

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch}/{N_EPOCHS-1}")
        print("-" * 50)

        loss, tissue_summary = train_one_epoch(classifier, cache, optimizer, epoch)

        # Save checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f'classifier_epoch_{epoch:03d}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': lizard_cp['config'],
            'classifier_type': 'ContextClassifier',
            'context_r': CONTEXT_R,
            'tissue_summary': {
                str(k): {
                    'mean_lymph_pct': float(np.mean(v['pred'])),
                    'mean_loss': float(np.mean(v['loss_values'])),
                    'total_cells': int(sum(v['n_cells'])),
                    'n_suppress': sum(1 for lt in v['loss_types'] if lt == 'suppress'),
                    'n_distill': sum(1 for lt in v['loss_types'] if lt.startswith('distill')),
                }
                for k, v in tissue_summary.items()
            },
        }, ckpt_path)

    print(f"\nTraining complete. Checkpoints saved to: {OUTPUT_DIR}")
    print(f"\nFinal Stage 1 checkpoint should be copied to checkpoints/context_soft_v4_epoch018.pth")
    print(f"(epoch 18 was selected as the Stage 2 init point in the original run)")


if __name__ == '__main__':
    main()
