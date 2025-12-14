"""
Compare Team Transformer vs. Custom TCN with Advanced Metrics:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- MPJPE (Mean Per Joint Position Error)

Run from root: python compare_metrics.py
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np

# Add subfolders to path so imports work
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import from the specific packages
from model import config as tf_config
from model import data as tf_data
from model.model import HandoverTransformer

# Import TCN config
from tcn import tcn_config
from tcn.model import TemporalConvNet

def load_transformer(device):
    path = tf_config.CKPT_PATH
    if not path.exists(): return None
    
    ckpt = torch.load(path, map_location=device)
    cfg_d_model = ckpt.get("cfg", {}).get("D_MODEL", tf_config.D_MODEL)
    
    model = HandoverTransformer(
        in_dim=ckpt["in_dim"],
        out_dim=ckpt["out_dim"],
        d_model=cfg_d_model,
        nhead=ckpt.get("cfg", {}).get("N_HEAD", tf_config.N_HEAD),
        nlayers=ckpt.get("cfg", {}).get("N_LAYERS", tf_config.N_LAYERS),
        ffdim=ckpt.get("cfg", {}).get("FFN_DIM", tf_config.FFN_DIM),
        dropout=ckpt.get("cfg", {}).get("DROPOUT", tf_config.DROPOUT),
        future_frames=tf_config.FUTURE_FRAMES,
    )
    state = ckpt["model"] if "model" in ckpt else ckpt["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()

def load_tcn(device):
    path = tcn_config.CKPT_DIR / "tcn_handover.pt"
    if not path.exists(): return None
    
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt['config']
    
    model = TemporalConvNet(
        in_dim=cfg['in_dim'],
        num_channels=cfg['num_channels'],
        out_dim=cfg['out_dim'],
        future_frames=tcn_config.FUTURE_FRAMES,
        kernel_size=cfg.get('kernel_size', 3)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()

def compute_mpjpe(pred, target):
    """
    Computes Mean Per Joint Position Error (Euclidean distance).
    pred/target shape: [Batch, Future_Frames, 63] (21 joints * 3 coords)
    """
    # Reshape to [Batch, Frames, 21, 3] to isolate x,y,z for each joint
    B, F, D = pred.shape
    n_joints = D // 3
    
    pred_3d = pred.view(B, F, n_joints, 3)
    target_3d = target.view(B, F, n_joints, 3)
    
    # Calculate Euclidean distance: sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
    diff = pred_3d - target_3d
    dist = torch.norm(diff, dim=3) # [Batch, Frames, 21]
    
    # Average over all joints, frames, and batches
    return dist.mean().item()

def compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing models on {device}...")
    
    transformer = load_transformer(device)
    tcn = load_tcn(device)
    
    if not transformer: print("❌ Transformer checkpoint missing."); return
    if not tcn: print("❌ TCN checkpoint missing."); return

    print("Loading test set...")
    _, _, test_loader = tf_data.build_loaders()
    if not test_loader: print("❌ Test set is empty."); return
    
    # Metrics Accumulators
    mse_crit = nn.MSELoss()
    mae_crit = nn.L1Loss()
    
    trans_mse, trans_mae, trans_mpjpe = 0.0, 0.0, 0.0
    tcn_mse, tcn_mae, tcn_mpjpe = 0.0, 0.0, 0.0
    
    n_batches = len(test_loader)
    print(f"Evaluating on {n_batches} test batches...")
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # --- TRANSFORMER ---
            pred_t = transformer(batch_x)
            trans_mse += mse_crit(pred_t, batch_y).item()
            trans_mae += mae_crit(pred_t, batch_y).item()
            trans_mpjpe += compute_mpjpe(pred_t, batch_y)
            
            # --- TCN ---
            pred_c = tcn(batch_x)
            tcn_mse += mse_crit(pred_c, batch_y).item()
            tcn_mae += mae_crit(pred_c, batch_y).item()
            tcn_mpjpe += compute_mpjpe(pred_c, batch_y)
            
    # Calculate Averages
    results = {
        "Transformer": {
            "MSE": trans_mse / n_batches,
            "MAE": trans_mae / n_batches,
            "MPJPE": trans_mpjpe / n_batches
        },
        "TCN": {
            "MSE": tcn_mse / n_batches,
            "MAE": tcn_mae / n_batches,
            "MPJPE": tcn_mpjpe / n_batches
        }
    }
    
    print("\n" + "="*45)
    print(f"{'METRIC':<10} | {'TRANSFORMER':<12} | {'TCN':<12} | {'WINNER':<10}")
    print("="*45)
    
    for metric in ["MSE", "MAE", "MPJPE"]:
        t_val = results["Transformer"][metric]
        c_val = results["TCN"][metric]
        diff = t_val - c_val
        
        if diff > 0:
            winner = "TCN 🏆"
            improv = f"(-{diff:.5f})"
        else:
            winner = "Transf."
            improv = f"(+{abs(diff):.5f})"
            
        print(f"{metric:<10} | {t_val:.6f}     | {c_val:.6f}     | {winner}")
        
    print("-" * 45)
    print("Note: MPJPE is the average Euclidean distance (error) per joint.")

if __name__ == "__main__":
    compare()
