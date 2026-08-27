"""
TCGA Embedding Extraction: WSI-crop context recovery for TCGA test ROIs.

Pipeline:
  1. For each ROI, crop 1024x1024 from the source WSI centered on the ROI's
     own coordinates (not the small pre-cut ROI PNG).
  2. Run CellViT-SAM-H on the WSI crop -> patch_tokens (1280, 64, 64).
  3. Load the stardist instance map -> map cells onto the token grid via
     the crop offset.
  4. Extract cell_token + context_token per cell, save to disk.

WSI crops provide real tissue context instead of zero-padding small patches:
the pre-cut ROI PNGs used for training-set extraction (see
train_stage1_context.py) are white-padded when undersized, which starves
edge cells of real context. Going back to the WSI and cropping a full
1024x1024 window centered on the ROI avoids that.

Classification (lizard / panoptils / Stage 1 / Stage 2) on the resulting
embeddings happens separately in classify_tcga.py, which reads the .pt
files this script writes.

Requires: openslide.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'cellvit_repo'))

import torch
import numpy as np
import re
import scipy.io as sio
from PIL import Image
from collections import defaultdict
import openslide


# ============================================================
# CONFIGURATION
# ============================================================

# External data — TIGER-style ROI-level annotations + TCGA WSIs.
# Point these at your own copies; not shipped in this repo.
# Set LOKI_OT_DATA_ROOT to wherever you've placed the TIGER WSIROIS data (see README "Dataset").
DATA_ROOT     = os.environ.get('LOKI_OT_DATA_ROOT', '/DataMount/xl260/wsirois')
BASE_DIR      = os.path.join(DATA_ROOT, 'roi-level-annotations', 'tissue-cells') + '/'
ROI_IMAGE_DIR = os.path.join(BASE_DIR, 'images/')
SELECTED_DIR  = os.path.join(BASE_DIR, 'selected_images/')
MAT_DIR       = os.path.join(BASE_DIR, 'selected_tcga_stardist_mat/')
MASK_DIR      = os.path.join(BASE_DIR, 'masks/')
WSI_DIR       = os.path.join(DATA_ROOT, 'wsi-level-annotations', 'images') + '/'

# Output: cached per-cell embeddings, consumed by classify_tcga.py
BASE_OUT      = os.path.join(BASE_DIR, 'selected_images_output/')
EMBED_OUT_DIR = os.path.join(BASE_OUT, 'cellvit_tcga_embeddings/')

CELLVIT_CHECKPOINT = os.path.join(REPO_ROOT, 'weights', 'sam_h', 'CellViT-SAM-H-x40-AMP.pth')

# Constants
DEVICE           = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE       = 1024
TOKEN_PATCH_SIZE = 16
TOKEN_GRID_SIZE  = 64   # PATCH_SIZE // TOKEN_PATCH_SIZE
CONTEXT_R        = 5    # expand cell bbox by 5 tokens (80px) in each direction
MIN_CELL_PIXELS  = 10


# ============================================================
# MODEL
# ============================================================

def load_cellvit_model():
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
    print("Loading CellViT-SAM-H model...")
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


# ============================================================
# ROI PARSING + WSI MAPPING
# ============================================================

def parse_roi_name(filename):
    """Parse full prefix and bbox from ROI filename.

    Example: 'TCGA-A1-A0SP-01Z-00-DX1.UUID_[3526, 26983, 3655, 27115].png'
    Returns: ('TCGA-A1-A0SP-01Z-00-DX1.UUID', (3526, 26983, 3655, 27115))
    """
    name = filename.replace('.png', '').replace('.mat', '')
    m = re.match(r'^(.+?)_\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]$', name)
    if not m:
        return None, None
    prefix = m.group(1)
    bbox = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return prefix, bbox


def build_wsi_lookup():
    """Build mapping from ROI prefix -> WSI .tif path."""
    lookup = {}
    for f in os.listdir(WSI_DIR):
        if f.endswith('.tif'):
            prefix = f.replace('.tif', '')
            lookup[prefix] = os.path.join(WSI_DIR, f)
    return lookup


def discover_tcga_rois(wsi_lookup):
    """Find non-selected TCGA ROIs that have .mat files and matching WSIs."""
    selected = set(os.listdir(SELECTED_DIR))

    all_images = sorted(f for f in os.listdir(ROI_IMAGE_DIR) if f.endswith('.png'))
    tcga_images = [f for f in all_images if f not in selected]
    print(f"Total ROIs: {len(all_images)}, selected: {len(selected)}, TCGA target: {len(tcga_images)}")

    # Group by WSI prefix (= patient), only include ROIs with .mat files
    patient_rois = defaultdict(list)
    n_missing_mat = 0
    n_missing_wsi = 0
    for img_file in tcga_images:
        roi_name = img_file.replace('.png', '')
        mat_path = os.path.join(MAT_DIR, f'{roi_name}.mat')
        if not os.path.exists(mat_path):
            n_missing_mat += 1
            continue
        prefix, bbox = parse_roi_name(img_file)
        if prefix is None:
            continue
        if prefix not in wsi_lookup:
            n_missing_wsi += 1
            continue
        patient_rois[prefix].append({
            'roi_name': roi_name,
            'img_file': img_file,
            'bbox': bbox,
        })

    total_rois = sum(len(v) for v in patient_rois.values())
    print(f"ROIs with .mat + WSI: {total_rois} across {len(patient_rois)} patients")
    if n_missing_mat > 0:
        print(f"  WARNING: {n_missing_mat} ROIs missing .mat files (skipped)")
    if n_missing_wsi > 0:
        print(f"  WARNING: {n_missing_wsi} ROIs missing WSI (skipped)")
    return patient_rois


# ============================================================
# TOKEN EXTRACTION
# ============================================================

def extract_cell_tokens(patch_tokens, inst_map, tissue_mask,
                        roi_x_in_crop, roi_y_in_crop):
    """Extract cell_token, context_token, tissue_type for each cell.

    inst_map: stardist instance map in small-patch coordinates (H_roi, W_roi)
    tissue_mask: tissue type mask in small-patch coordinates (H_roi, W_roi)
    roi_x_in_crop, roi_y_in_crop: offset from small patch to 1024 crop
    patch_tokens: CellViT tokens (C, 64, 64)

    Returns: cell_ids, cell_tokens (N,C), context_tokens (N,C), cell_tissue (N,), centroids dict
    """
    C = patch_tokens.shape[0]
    cell_ids = []
    cell_tokens_list = []
    context_tokens_list = []
    cell_tissue_list = []
    centroids = {}

    for cid in np.unique(inst_map):
        if cid == 0:
            continue
        mask = inst_map == cid
        if mask.sum() < MIN_CELL_PIXELS:
            continue

        ys, xs = np.where(mask)

        # Centroid in small-patch coords (for JSON/plot output)
        cy_small, cx_small = float(ys.mean()), float(xs.mean())

        # Tissue type from ROI-level mask at cell centroid
        ty_mask = min(int(cy_small), tissue_mask.shape[0] - 1)
        tx_mask = min(int(cx_small), tissue_mask.shape[1] - 1)
        tt = int(tissue_mask[ty_mask, tx_mask])

        # Offset to 1024 crop coordinates
        ys_crop = ys + roi_y_in_crop
        xs_crop = xs + roi_x_in_crop

        # Cell token bbox in token grid
        ty0 = max(0, int(np.floor(ys_crop.min() / TOKEN_PATCH_SIZE)))
        ty1 = min(TOKEN_GRID_SIZE, int(np.ceil((ys_crop.max() + 1) / TOKEN_PATCH_SIZE)))
        tx0 = max(0, int(np.floor(xs_crop.min() / TOKEN_PATCH_SIZE)))
        tx1 = min(TOKEN_GRID_SIZE, int(np.ceil((xs_crop.max() + 1) / TOKEN_PATCH_SIZE)))
        if ty1 <= ty0 or tx1 <= tx0:
            continue

        ct = patch_tokens[:, ty0:ty1, tx0:tx1].reshape(C, -1).mean(dim=1)

        # Context token (expanded bbox)
        ey0 = max(0, ty0 - CONTEXT_R)
        ey1 = min(TOKEN_GRID_SIZE, ty1 + CONTEXT_R)
        ex0 = max(0, tx0 - CONTEXT_R)
        ex1 = min(TOKEN_GRID_SIZE, tx1 + CONTEXT_R)
        ctx = patch_tokens[:, ey0:ey1, ex0:ex1].reshape(C, -1).mean(dim=1)

        cell_ids.append(int(cid))
        cell_tokens_list.append(ct)
        context_tokens_list.append(ctx)
        cell_tissue_list.append(tt)
        centroids[int(cid)] = [cy_small, cx_small]

    if len(cell_tokens_list) == 0:
        return [], None, None, None, {}

    cell_tokens = torch.stack(cell_tokens_list)       # (N, C)
    context_tokens = torch.stack(context_tokens_list)  # (N, C)
    cell_tissue = torch.tensor(cell_tissue_list)       # (N,)
    return cell_ids, cell_tokens, context_tokens, cell_tissue, centroids


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(EMBED_OUT_DIR, exist_ok=True)

    print("Loading model...")
    cellvit_model = load_cellvit_model()

    print("\nBuilding WSI lookup...")
    wsi_lookup = build_wsi_lookup()
    print(f"  {len(wsi_lookup)} WSI files indexed")

    print("\nDiscovering TCGA ROIs...")
    patient_rois = discover_tcga_rois(wsi_lookup)

    patients = sorted(patient_rois.keys())
    n_saved = 0
    n_skipped = 0

    print(f"\n{'='*80}")
    print(f"Extracting embeddings for {sum(len(v) for v in patient_rois.values())} "
          f"ROIs across {len(patients)} patients")
    print(f"{'='*80}")

    for pi, wsi_prefix in enumerate(patients):
        rois = patient_rois[wsi_prefix]
        wsi_path = wsi_lookup[wsi_prefix]

        slide = openslide.OpenSlide(wsi_path)
        wsi_w, wsi_h = slide.dimensions

        for roi_info in rois:
            roi_name = roi_info['roi_name']
            bbox     = roi_info['bbox']
            x1, y1, x2, y2 = bbox
            roi_w, roi_h = x2 - x1, y2 - y1

            # Compute 1024x1024 WSI crop centered on ROI
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop_x0 = max(0, min(cx - PATCH_SIZE // 2, wsi_w - PATCH_SIZE))
            crop_y0 = max(0, min(cy - PATCH_SIZE // 2, wsi_h - PATCH_SIZE))
            roi_x_in_crop = x1 - crop_x0
            roi_y_in_crop = y1 - crop_y0

            # Read 1024x1024 crop from WSI
            region   = slide.read_region((crop_x0, crop_y0), 0, (PATCH_SIZE, PATCH_SIZE))
            crop_rgb = np.array(region.convert('RGB'))

            # CellViT forward on WSI crop
            img_norm = (crop_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
            img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    preds = cellvit_model(img_t, retrieve_tokens=True)
            patch_tokens = preds['tokens'].cpu()[0]  # (1280, 64, 64)

            # Load stardist inst_map (small-patch coords)
            mat_path = os.path.join(MAT_DIR, f'{roi_name}.mat')
            inst_map = sio.loadmat(mat_path)['inst_map']

            # Crop inst_map to match ROI image size (stardist may differ by 1-2 px)
            inst_map = inst_map[:roi_h, :roi_w]

            # Load ROI-level tissue mask
            mask_path = os.path.join(MASK_DIR, f'{roi_name}.png')
            if os.path.exists(mask_path):
                tissue_mask = np.array(Image.open(mask_path))
            else:
                tissue_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

            # Extract cell tokens with offset mapping
            cell_ids, cell_tokens, context_tokens, cell_tissue, centroids = \
                extract_cell_tokens(patch_tokens, inst_map, tissue_mask,
                                    roi_x_in_crop, roi_y_in_crop)

            if len(cell_ids) == 0:
                n_skipped += 1
                continue

            # Save cell/context embeddings as .pt (consumed by classify_tcga.py)
            embed_path = os.path.join(EMBED_OUT_DIR, f'{roi_name}.pt')
            torch.save({
                'cell_tokens':    cell_tokens,       # (N, 1280)
                'context_tokens': context_tokens,    # (N, 1280)
                'cell_ids':       cell_ids,          # list of int
                'cell_tissue':    cell_tissue,       # (N,)
                'centroids':      centroids,         # dict {cid: [cy, cx]}
                'bbox_wsi':       list(bbox),
                'crop_origin':    [crop_x0, crop_y0],
                'roi_offset_in_crop': [roi_x_in_crop, roi_y_in_crop],
            }, embed_path)
            n_saved += 1

            print(f"  {roi_name:<75s} {len(cell_ids):5d} cells")

        slide.close()

        if (pi + 1) % 10 == 0:
            print(f"  --- {pi+1}/{len(patients)} patients done, {n_saved} ROIs embedded ---")

    print(f"\nDone.")
    print(f"  Embeddings ({n_saved} files): {EMBED_OUT_DIR}")
    if n_skipped > 0:
        print(f"  Skipped: {n_skipped}")


if __name__ == '__main__':
    main()
