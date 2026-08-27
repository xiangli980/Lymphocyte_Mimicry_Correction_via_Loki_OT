"""
Loki-OT Stage 2: Build UOT Teacher Labels

For each tissue-ROI group, solves an unbalanced-optimal-transport (Sinkhorn)
problem whose target marginal is the MLLM lymphocyte-percentage estimate,
using a cost matrix built from the Stage 1 (context_soft_v4) baseline's own
softmax + a context-cosine penalty in "strict zones" (tumor/T3/T4).

OT is only applied when the baseline over-predicts lymphocytes by
>= SWITCH_THRESHOLD. T4 and T5 are excluded from OT entirely (OT was
counter-productive / ill-conditioned there) and instead fall back to the
Stage 1 classifier's own predictions.

Output:
  {roi_name: {'teacher_soft': tensor(N, 6)}}
  saved to TEACHER_PATH
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_PATH   = os.path.join(REPO_ROOT, 'outputs', 'cache', 'cached_cell_tokens.pt')
MLLM_JSON    = os.path.join(REPO_ROOT, 'data', 'mllm_reference.json')
TEACHER_PATH = os.path.join(REPO_ROOT, 'outputs', 'cache', 'uot_teacher.pt')

STAGE1_CHECKPOINT = os.path.join(REPO_ROOT, 'checkpoints', 'context_soft_v4_epoch018.pth')
LIZARD_CHECKPOINT = os.path.join(REPO_ROOT, 'cellvit_repo', 'checkpoints', 'classifier', 'sam-h', 'lizard.pth')

DEVICE = torch.device('cpu')

LYMPH_CLASS_IDX = 2

# UOT hyperparameters
GAMMA          = 0.1
REG_M          = 3.0
SINKHORN_ITERS = 100
SINKHORN_TOL   = 1e-6
STRICT_ZONES      = {1, 3, 4}
CONTEXT_ALPHA     = 1.0
SWITCH_THRESHOLD  = 0.10

# Skip OT for T4 and T5, fall back to the Stage 1 classifier's own predictions
OT_SKIP_ZONES = {4, 5}
# T4: OT counter-productive (1.3x selectivity vs Stage 1's 2.8x); Stage 1 preserves context signal
# T5: OT ill-conditioned (MLLM ~0%, selectivity = 1.0x = random suppression)

TISSUE_NAMES = {1: 'T1 (tumor)', 2: 'T2 (immune)', 3: 'T3', 4: 'T4 (glands)',
                5: 'T5 (bg)', 6: 'T6 (stroma)', 7: 'T7 (necrosis)'}


# ============================================================
# MODEL (for Stage 1 inference)
# ============================================================

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


def load_stage1_classifier():
    """Load the Stage 1 ContextClassifier for computing its predictions."""
    lizard_cp = torch.load(LIZARD_CHECKPOINT, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')
    hidden_dim = lizard_conf['model'].get('hidden_dim', 100)
    num_classes = lizard_conf['data']['num_classes']

    classifier = ContextClassifier(
        cell_dim=1280, context_dim=1280,
        hidden_dim=hidden_dim, num_classes=num_classes, drop_rate=0
    )

    stage1_cp = torch.load(STAGE1_CHECKPOINT, map_location='cpu', weights_only=False)
    classifier.load_state_dict(stage1_cp['model_state_dict'])
    classifier.eval()
    print(f"  Loaded Stage 1 classifier from {STAGE1_CHECKPOINT} (epoch {stage1_cp.get('epoch', '?')})")
    return classifier


# ============================================================
# UOT SOLVER
# ============================================================

def uot_sinkhorn(cost, mu, nu, reg, reg_m, max_iters=100, tol=1e-6):
    N, K = cost.shape
    fi = reg_m / (reg_m + reg)
    log_K = -cost / reg
    log_mu = torch.log(mu.clamp(min=1e-30))
    log_nu = torch.log(nu.clamp(min=1e-30))

    u = torch.zeros(N, dtype=cost.dtype, device=cost.device)
    v = torch.zeros(K, dtype=cost.dtype, device=cost.device)

    for it in range(max_iters):
        u_prev = u.clone()
        u = fi * (log_mu - torch.logsumexp(log_K + v.unsqueeze(0), dim=1))
        v = fi * (log_nu - torch.logsumexp(log_K + u.unsqueeze(1), dim=0))
        if (it + 1) % 10 == 0:
            if torch.max(torch.abs(u - u_prev)) < tol:
                break

    log_P = u.unsqueeze(1) + log_K + v.unsqueeze(0)
    return torch.exp(log_P)


# ============================================================
# COST FUNCTIONS
# ============================================================

def get_context_penalty(cell_tokens, context_tokens):
    raw_sim = F.cosine_similarity(cell_tokens.float(), context_tokens.float(), dim=1)
    return F.relu(raw_sim) ** 2


def build_prob_cost_matrix(baseline_soft, lymph_idx=2):
    p_lymph = baseline_soft[:, lymph_idx]
    return torch.stack([p_lymph, 1.0 - p_lymph], dim=1)


def build_context_cost_matrix(baseline_soft, cell_tokens, context_tokens,
                               context_alpha, lymph_idx=2):
    p_lymph = baseline_soft[:, lymph_idx]
    penalty = get_context_penalty(cell_tokens, context_tokens)
    c_non = p_lymph
    c_lym = (1.0 - p_lymph) + context_alpha * penalty
    return torch.stack([c_non, c_lym], dim=1)


# ============================================================
# TEACHER CONSTRUCTION
# ============================================================

def build_teacher_for_roi(roi_data, mllm_estimates, stage1_soft):
    """Build per-cell 6-class teacher soft labels for one ROI.

    - OT_SKIP_ZONES = {4, 5}: fall back to stage1_soft (not the lizard baseline).
    - OT itself uses lizard_soft for the cost matrix and the switch gate.

    Returns:
        teacher_soft: (N, 6) float tensor, sums to 1 per row
        n_ot_tissues: number of tissues where OT was applied
        n_skip_tissues: number of tissues skipped by OT_SKIP_ZONES
    """
    lizard_soft    = roi_data['lizard_soft'].float()
    cell_tissue    = roi_data['cell_tissue']
    cell_tokens    = roi_data['cell_tokens'].float()
    context_tokens = roi_data['context_tokens'].float()
    N = lizard_soft.shape[0]

    # Init from lizard_soft; T4/T5 cells get overwritten with stage1_soft below
    teacher_soft = lizard_soft.clone()
    lizard_classes = lizard_soft.argmax(dim=1)
    n_ot_tissues = 0
    n_skip_tissues = 0

    for tissue_type in cell_tissue.unique().tolist():
        if tissue_type == 0:
            continue

        if tissue_type in OT_SKIP_ZONES:
            tissue_mask = (cell_tissue == tissue_type)
            teacher_soft[tissue_mask] = stage1_soft[tissue_mask]
            n_skip_tissues += 1
            continue

        tissue_mask = (cell_tissue == tissue_type)
        n_tissue = tissue_mask.sum().item()
        if n_tissue < 2:
            continue

        tissue_str = str(tissue_type)
        if tissue_str not in mllm_estimates:
            continue

        mllm_pct = mllm_estimates[tissue_str] / 100.0

        tissue_liz_classes = lizard_classes[tissue_mask]
        n_liz_lymph = (tissue_liz_classes == LYMPH_CLASS_IDX).sum().item()
        liz_lymph_pct = n_liz_lymph / n_tissue

        # Gate: only apply OT when lizard over-predicts by >= SWITCH_THRESHOLD
        if liz_lymph_pct - mllm_pct < SWITCH_THRESHOLD:
            continue

        # Build cost matrix (using lizard_soft)
        tissue_soft  = lizard_soft[tissue_mask]
        tissue_cells = cell_tokens[tissue_mask]
        tissue_ctx   = context_tokens[tissue_mask]

        is_strict = tissue_type in STRICT_ZONES
        if is_strict and CONTEXT_ALPHA > 0:
            cost = build_context_cost_matrix(
                tissue_soft, tissue_cells, tissue_ctx, CONTEXT_ALPHA, LYMPH_CLASS_IDX
            )
        else:
            cost = build_prob_cost_matrix(tissue_soft, LYMPH_CLASS_IDX)

        mu = torch.ones(n_tissue, dtype=torch.float32) / n_tissue
        nu = torch.tensor([1.0 - mllm_pct, mllm_pct], dtype=torch.float32)
        nu = nu.clamp(min=1e-8)
        nu = nu / nu.sum()

        # Run UOT
        P = uot_sinkhorn(cost, mu, nu, GAMMA, REG_M, SINKHORN_ITERS, SINKHORN_TOL)

        row_sum = P.sum(dim=1, keepdim=True).clamp(min=1e-12)
        P_norm  = P / row_sum
        p_lymph = P_norm[:, 1]

        # Distribute non-lymph mass proportional to lizard_soft (excl. lymph column)
        liz_no_lymph = tissue_soft.clone()
        liz_no_lymph[:, LYMPH_CLASS_IDX] = 0.0
        row_sum_nl = liz_no_lymph.sum(dim=1, keepdim=True).clamp(min=1e-12)
        liz_no_lymph = liz_no_lymph / row_sum_nl

        t = liz_no_lymph * (1.0 - p_lymph).unsqueeze(1)
        t[:, LYMPH_CLASS_IDX] = p_lymph

        teacher_soft[tissue_mask] = t
        n_ot_tissues += 1

    return teacher_soft, n_ot_tissues, n_skip_tissues


# ============================================================
# MLLM LOADING
# ============================================================

def attach_mllm_estimates(cache):
    print("Loading MLLM reference...")
    with open(MLLM_JSON, 'r') as f:
        mllm_all = json.load(f)

    n_with_mllm = 0
    for roi_name in cache:
        key = roi_name + '.png'
        est = mllm_all.get(key, {})
        cache[roi_name]['mllm_estimates'] = est
        if est:
            n_with_mllm += 1

    print(f"  {n_with_mllm}/{len(cache)} ROIs have MLLM estimates")
    return cache


# ============================================================
# MAIN
# ============================================================

def main():
    fi = REG_M / (REG_M + GAMMA)
    print(f"Building UOT teacher labels")
    print(f"  gamma={GAMMA}, reg_m={REG_M}, fi={fi:.3f}, context_alpha={CONTEXT_ALPHA}")
    print(f"  Strict zones: {sorted(STRICT_ZONES)}")
    print(f"  OT skip zones: {sorted(OT_SKIP_ZONES)} (T4+T5 skipped; fall back to Stage 1 predictions)")
    print(f"  Switch threshold: {SWITCH_THRESHOLD}")
    print(f"  Cache: {CACHE_PATH}")
    print(f"  Output: {TEACHER_PATH}")

    print(f"\nLoading cache...")
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    total_cells = sum(v['cell_tokens'].shape[0] for v in cache.values())
    print(f"  {len(cache)} ROIs, {total_cells} cells")

    cache = attach_mllm_estimates(cache)

    # Load Stage 1 classifier for T4/T5 fallback
    print(f"\nLoading Stage 1 classifier for T4/T5 fallback...")
    stage1_clf = load_stage1_classifier()

    print(f"\nBuilding teacher labels...")
    teacher_cache = {}
    n_total_ot = 0
    n_total_skip = 0
    n_total_roi = len(cache)

    for idx, (roi_name, roi_data) in enumerate(sorted(cache.items())):
        mllm_estimates = roi_data.get('mllm_estimates', {})

        # Compute Stage 1 predictions for this ROI
        with torch.no_grad():
            cell_t = roi_data['cell_tokens'].float()
            ctx_t  = roi_data['context_tokens'].float()
            stage1_logits = stage1_clf(cell_t, ctx_t)
            stage1_soft = F.softmax(stage1_logits, dim=1)

        teacher_soft, n_ot, n_skip = build_teacher_for_roi(roi_data, mllm_estimates, stage1_soft)
        teacher_cache[roi_name] = {'teacher_soft': teacher_soft}
        n_total_ot += n_ot
        n_total_skip += n_skip

        if (idx + 1) % 30 == 0:
            print(f"  ... {idx+1}/{n_total_roi} ROIs processed")

    print(f"\n  Done. OT applied to {n_total_ot} tissue-ROI pairs")
    print(f"  Skipped (OT_SKIP_ZONES): {n_total_skip} tissue-ROI pairs (T4+T5 -> Stage 1 predictions)")

    # Sanity check
    all_ok = True
    for roi_name, t in teacher_cache.items():
        ts = t['teacher_soft']
        row_sums = ts.sum(dim=1)
        if ts.min() < -1e-6 or (row_sums - 1.0).abs().max() > 1e-4:
            print(f"  WARN: {roi_name} teacher sanity check failed "
                  f"(min={ts.min():.4f}, max_row_err={(row_sums-1).abs().max():.6f})")
            all_ok = False
    if all_ok:
        print("  Sanity check passed: all teacher rows sum to 1, all non-negative.")

    # Summary: how different is teacher from lizard_soft?
    mean_l1_diffs = []
    for roi_name, t in teacher_cache.items():
        ts  = t['teacher_soft']
        liz = cache[roi_name]['lizard_soft'].float()
        diff = (ts - liz).abs().mean().item()
        mean_l1_diffs.append(diff)
    print(f"\n  Mean per-cell L1(teacher, lizard): {np.mean(mean_l1_diffs)*100:.2f}%")

    os.makedirs(os.path.dirname(TEACHER_PATH), exist_ok=True)
    torch.save({
        'gamma': GAMMA,
        'reg_m': REG_M,
        'context_alpha': CONTEXT_ALPHA,
        'strict_zones': sorted(STRICT_ZONES),
        'switch_threshold': SWITCH_THRESHOLD,
        'ot_skip_zones': sorted(OT_SKIP_ZONES),
        'teacher': teacher_cache,
    }, TEACHER_PATH)
    print(f"\nTeacher labels saved to {TEACHER_PATH}")


if __name__ == '__main__':
    main()
