"""
Loki-OT Stage 2: Distillation Training

Fine-tunes a fresh ContextClassifier, reinitialized from the Stage 1
(context_soft_v4) checkpoint, against the UOT teacher labels built by
build_uot_teacher.py. Loss mixes a temperature-scaled KL against the soft
teacher with a cross-entropy against the teacher's hard argmax.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# CONFIGURATION  (update paths for your container)
# ============================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_PATH        = os.path.join(REPO_ROOT, 'outputs', 'cache', 'cached_cell_tokens.pt')
TEACHER_PATH      = os.path.join(REPO_ROOT, 'outputs', 'cache', 'uot_teacher.pt')
STAGE1_CHECKPOINT = os.path.join(REPO_ROOT, 'checkpoints', 'context_soft_v4_epoch018.pth')
LIZARD_CHECKPOINT = os.path.join(REPO_ROOT, 'cellvit_repo', 'checkpoints', 'classifier', 'sam-h', 'lizard.pth')
OUTPUT_DIR        = os.path.join(REPO_ROOT, 'outputs', 'stage2_distill')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Stage 2 training hyperparameters
LR       = 1e-5    # 10x lower than Stage 1 — gentle fine-tune from Stage 1 init
N_EPOCHS = 5        # preserve Stage 1 prior
ALPHA    = 0.5      # loss mixing: alpha * KL_soft + (1 - alpha) * CE_hard
TEMP     = 4.0      # KL temperature — learn ranking, not absolute scale

LYMPH_CLASS_IDX = 2

TISSUE_NAMES = {1: 'T1 (tumor)', 2: 'T2 (immune)', 3: 'T3', 4: 'T4 (glands)',
                5: 'T5 (bg)', 6: 'T6 (stroma)', 7: 'T7 (necrosis)'}


# ============================================================
# MODEL  (identical architecture to Stage 1)
# ============================================================

class LinearClassifier(nn.Module):
    """Lizard baseline classifier (for weight extraction during init)."""
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
    """Context-aware cell classifier (identical to Stage 1).

    Concatenates cell token (1280) with spatial context token (1280) -> 2560-dim input.
    """
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


def load_stage1_checkpoint(stage1_path, lizard_path):
    """Load the Stage 1 ContextClassifier checkpoint.

    Returns classifier with Stage 1 weights loaded, plus lizard config for metadata.
    """
    lizard_cp = torch.load(lizard_path, map_location='cpu', weights_only=False)
    lizard_conf = unflatten_dict(lizard_cp['config'], '.')
    hidden_dim = lizard_conf['model'].get('hidden_dim', 100)
    num_classes = lizard_conf['data']['num_classes']

    classifier = ContextClassifier(
        cell_dim=1280, context_dim=1280,
        hidden_dim=hidden_dim, num_classes=num_classes, drop_rate=0
    )

    stage1_cp = torch.load(stage1_path, map_location='cpu', weights_only=False)
    classifier.load_state_dict(stage1_cp['model_state_dict'])
    print(f"  Loaded Stage 1 weights from {stage1_path}")
    if 'epoch' in stage1_cp:
        print(f"  Stage 1 checkpoint epoch: {stage1_cp['epoch']}")

    return classifier, lizard_cp, hidden_dim, num_classes


# ============================================================
# LOSS  (temperature KL + hard CE)
# ============================================================

def kl_distill_loss_tempered(student_logits, teacher_soft, T=4.0):
    """Temperature-scaled KL divergence: KL(student/T || teacher^(1/T)).

    At T=1 this reduces to standard KL(student || teacher). At T>1 both
    distributions are softened, forcing the student to learn relative
    ranking rather than absolute probability values.

    Args:
        student_logits: (N, 6) raw logits from ContextClassifier
        teacher_soft:   (N, 6) soft teacher distribution (sums to 1 per row)
        T:              temperature (default 4.0)

    Returns:
        loss: scalar tensor, scaled by T^2 (Hinton et al. 2015)
    """
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    # Raise teacher probs to power 1/T and renormalize
    teacher_T = teacher_soft.clamp(min=1e-8).pow(1.0 / T)
    teacher_T = teacher_T / teacher_T.sum(dim=1, keepdim=True)
    return T * T * F.kl_div(student_log_soft, teacher_T, reduction='batchmean')


def mixed_loss(student_logits, teacher_soft, alpha=0.5, T=4.0):
    """Mixed loss: alpha * KL_soft(T) + (1 - alpha) * CE_hard.

    The KL component (with temperature) teaches relative cell ordering from
    the OT teacher. The CE component anchors a globally consistent decision
    boundary across tissues by training against the teacher's hard argmax.

    Args:
        student_logits: (N, 6) raw logits
        teacher_soft:   (N, 6) soft teacher distribution
        alpha:          mixing weight (default 0.5)
        T:              KL temperature (default 4.0)

    Returns:
        loss: scalar tensor
        kl_loss: scalar (for logging)
        ce_loss: scalar (for logging)
    """
    kl = kl_distill_loss_tempered(student_logits, teacher_soft, T)
    teacher_hard = teacher_soft.argmax(dim=1)
    ce = F.cross_entropy(student_logits, teacher_hard)
    return alpha * kl + (1 - alpha) * ce, kl.item(), ce.item()


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(classifier, cache, teacher_cache, optimizer, epoch):
    classifier.train()
    epoch_losses = []
    epoch_kl = []
    epoch_ce = []
    lymph_pcts_by_tissue = {}

    roi_names = list(cache.keys())
    roi_order = np.random.permutation(len(roi_names))

    for idx in roi_order:
        roi_name = roi_names[idx]
        roi = cache[roi_name]

        if roi_name not in teacher_cache:
            continue

        cell_tokens    = roi['cell_tokens'].float().to(DEVICE)
        context_tokens = roi['context_tokens'].float().to(DEVICE)
        cell_tissue    = roi['cell_tissue']
        teacher_soft   = teacher_cache[roi_name]['teacher_soft'].to(DEVICE)

        optimizer.zero_grad()

        logits = classifier(cell_tokens, context_tokens)
        loss, kl_val, ce_val = mixed_loss(logits, teacher_soft, ALPHA, TEMP)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_kl.append(kl_val)
        epoch_ce.append(ce_val)

        with torch.no_grad():
            pred_classes = logits.argmax(dim=1).cpu()
            for t in cell_tissue.unique().tolist():
                if t == 0:
                    continue
                mask = (cell_tissue == t)
                pct = (pred_classes[mask] == LYMPH_CLASS_IDX).float().mean().item()
                lymph_pcts_by_tissue.setdefault(t, []).append(pct)

    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    avg_kl = float(np.mean(epoch_kl)) if epoch_kl else 0.0
    avg_ce = float(np.mean(epoch_ce)) if epoch_ce else 0.0

    print(f"  Loss: {avg_loss:.6f}  (KL={avg_kl:.6f}, CE={avg_ce:.6f})")
    print(f"  {'Tissue':<16} | {'Pred lymph%':>12} | {'Teacher lymph%':>15}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*15}")

    teacher_lymph_by_tissue = {}
    for roi_name, t_data in teacher_cache.items():
        if roi_name not in cache:
            continue
        ct = cache[roi_name]['cell_tissue']
        ts = t_data['teacher_soft']
        for t in ct.unique().tolist():
            if t == 0:
                continue
            mask = (ct == t)
            pct = ts[mask, LYMPH_CLASS_IDX].mean().item()
            teacher_lymph_by_tissue.setdefault(t, []).append(pct)

    for t in sorted(lymph_pcts_by_tissue.keys()):
        t_name = TISSUE_NAMES.get(t, f'T{t}')
        pred_mean = np.mean(lymph_pcts_by_tissue[t]) * 100
        teach_mean = np.mean(teacher_lymph_by_tissue.get(t, [0.0])) * 100
        print(f"  {t_name:<16} | {pred_mean:11.1f}% | {teach_mean:14.1f}%")

    return avg_loss, lymph_pcts_by_tissue


# ============================================================
# EVALUATION
# ============================================================

def evaluate(classifier, cache, teacher_cache):
    classifier.eval()
    tissue_pred = {}
    tissue_teach = {}
    tissue_liz = {}

    with torch.no_grad():
        for roi_name, roi in cache.items():
            cell_tokens    = roi['cell_tokens'].float().to(DEVICE)
            context_tokens = roi['context_tokens'].float().to(DEVICE)
            cell_tissue    = roi['cell_tissue']
            lizard_soft    = roi['lizard_soft']

            logits = classifier(cell_tokens, context_tokens)
            pred_classes  = logits.argmax(dim=1).cpu()
            liz_classes   = lizard_soft.argmax(dim=1)

            teacher_soft = teacher_cache.get(roi_name, {}).get('teacher_soft')

            for t in cell_tissue.unique().tolist():
                if t == 0:
                    continue
                mask = (cell_tissue == t)
                pred_pct = (pred_classes[mask] == LYMPH_CLASS_IDX).float().mean().item()
                liz_pct  = (liz_classes[mask]  == LYMPH_CLASS_IDX).float().mean().item()
                tissue_pred.setdefault(t, []).append(pred_pct)
                tissue_liz.setdefault(t, []).append(liz_pct)
                if teacher_soft is not None:
                    teach_pct = teacher_soft[mask, LYMPH_CLASS_IDX].mean().item()
                    tissue_teach.setdefault(t, []).append(teach_pct)

    print(f"\n  {'Tissue':<16} | {'Lizard':>8} | {'Teacher':>8} | {'Stage2':>8}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for t in sorted(tissue_pred.keys()):
        t_name = TISSUE_NAMES.get(t, f'T{t}')
        liz  = np.mean(tissue_liz.get(t, [0])) * 100
        pred = np.mean(tissue_pred[t]) * 100
        teach = np.mean(tissue_teach.get(t, [0])) * 100
        print(f"  {t_name:<16} | {liz:7.1f}% | {teach:7.1f}% | {pred:7.1f}%")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loki-OT Stage 2: UOT-distilled classifier (Stage 1 init + T4/T5 fallback)")
    print(f"  Loss: {ALPHA} * KL(T={TEMP}) + {1-ALPHA} * CE(hard)")
    print(f"  LR={LR}, epochs={N_EPOCHS}")
    print(f"  Init: Stage 1 checkpoint")

    # -- Load cache ------------------------------------------------
    print(f"\nLoading cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    total_cells = sum(v['cell_tokens'].shape[0] for v in cache.values())
    print(f"  {len(cache)} ROIs, {total_cells} cells")

    # -- Load teacher ------------------------------------------
    print(f"\nLoading teacher from {TEACHER_PATH}...")
    teacher_data = torch.load(TEACHER_PATH, map_location='cpu', weights_only=False)
    teacher_cache = teacher_data['teacher']
    print(f"  {len(teacher_cache)} ROIs with teacher labels")
    print(f"  Built with: gamma={teacher_data.get('gamma')}, reg_m={teacher_data.get('reg_m')}, "
          f"context_alpha={teacher_data.get('context_alpha')}")
    if 'ot_skip_zones' in teacher_data:
        print(f"  OT skip zones: {teacher_data['ot_skip_zones']} (T4+T5 use Stage 1 predictions)")

    # -- Tissue distribution ---------------------------------------
    tissue_counts = {}
    for v in cache.values():
        for t in v['cell_tissue'].unique().tolist():
            if t == 0:
                continue
            tissue_counts[t] = tissue_counts.get(t, 0) + (v['cell_tissue'] == t).sum().item()
    print("\nTissue distribution:")
    for t in sorted(tissue_counts):
        print(f"  {TISSUE_NAMES.get(t, f'T{t}')}: {tissue_counts[t]} cells")

    # -- Initialize from Stage 1 checkpoint -----------------------------
    print(f"\nInitializing ContextClassifier from Stage 1 checkpoint...")
    classifier, lizard_cp, hidden_dim, num_classes = load_stage1_checkpoint(
        STAGE1_CHECKPOINT, LIZARD_CHECKPOINT
    )
    classifier.to(DEVICE)
    trainable = sum(p.numel() for p in classifier.parameters())
    print(f"  Architecture: [1280+1280] -> {hidden_dim} -> {num_classes}")
    print(f"  Trainable params: {trainable:,}")

    # -- Verify Stage 1 init: should NOT match lizard (Stage 1 has been trained) --
    print("  Verifying Stage 1 init differs from lizard baseline...")
    classifier.eval()
    sample_roi = list(cache.values())[0]
    with torch.no_grad():
        s_cell = sample_roi['cell_tokens'][:5].float().to(DEVICE)
        s_ctx  = sample_roi['context_tokens'][:5].float().to(DEVICE)
        stage2_out = classifier(s_cell, s_ctx)

        liz_clf_tmp = LinearClassifier(embed_dim=1280, hidden_dim=hidden_dim,
                                       num_classes=num_classes, drop_rate=0)
        liz_clf_tmp.load_state_dict(lizard_cp['model_state_dict'])
        liz_clf_tmp.eval().to(DEVICE)
        liz_out = liz_clf_tmp(s_cell)
        del liz_clf_tmp

    max_diff = (stage2_out - liz_out).abs().max().item()
    print(f"  Max diff from lizard: {max_diff:.4f} (should be >0, Stage 1 was trained)")

    # -- Pre-training eval (Stage 1 baseline) ---------------------------
    print("\n-- Pre-training evaluation (Stage 1 weights) --")
    evaluate(classifier, cache, teacher_cache)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    print(f"\n{'='*70}")
    print(f"Stage 2 Training: {ALPHA}*KL(T={TEMP}) + {1-ALPHA}*CE(hard), from Stage 1 init")
    print(f"{'='*70}")

    best_loss = float('inf')
    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch}/{N_EPOCHS-1}")
        print("-" * 50)

        avg_loss, lymph_pcts = train_one_epoch(
            classifier, cache, teacher_cache, optimizer, epoch
        )

        # Checkpoint every epoch
        ckpt_path = os.path.join(OUTPUT_DIR, f'classifier_epoch_{epoch:03d}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': lizard_cp['config'],
            'classifier_type': 'ContextClassifier_stage2',
            'loss_type': 'mixed_kl_ce',
            'loss_alpha': ALPHA,
            'loss_temp': TEMP,
            'init_from': 'stage1',
        }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(OUTPUT_DIR, 'classifier_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': lizard_cp['config'],
                'classifier_type': 'ContextClassifier_stage2',
                'loss_type': 'mixed_kl_ce',
                'loss_alpha': ALPHA,
                'loss_temp': TEMP,
                'init_from': 'stage1',
            }, best_path)
            print(f"  -> Best model saved (loss={avg_loss:.6f})")

        # Eval every epoch (only 5 epochs, want to see all)
        evaluate(classifier, cache, teacher_cache)

    print(f"\nTraining complete. Checkpoints saved to: {OUTPUT_DIR}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"\nFinal checkpoint (epoch 3 in the original run) should be copied to "
          f"checkpoints/loki_ot_v15_epoch003.pth")


if __name__ == '__main__':
    main()
