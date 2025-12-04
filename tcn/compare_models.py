"""
Compare Team Transformer vs. Custom TCN
Run from root: python compare_models.py
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add subfolders to path so imports work
sys.path.append(str(Path.cwd()))

# Import from the specific packages
from team_transformer import config as tf_config
from team_transformer import data as tf_data
from team_transformer.model import HandoverTransformer

# --- FIX: Import tcn_config ---
from tcn import tcn_config     # Was 'config'
from tcn.model import TemporalConvNet
# ------------------------------

def load_transformer(device):
    path = tf_config.CKPT_PATH
    if not path.exists(): return None
    
    ckpt = torch.load(path, map_location=device)
    # Handle team's config structure
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
    # Use tcn_config for path
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

def compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing models on {device}...")
    
    # Load Models
    transformer = load_transformer(device)
    tcn = load_tcn(device)
    
    if not transformer: print("❌ Transformer checkpoint missing. Run: python -m team_transformer.train"); return
    if not tcn: print("❌ TCN checkpoint missing. Run: python -m tcn.train"); return

    # Load Test Data (Using Team's data loader as the standard)
    print("Loading test set...")
    _, _, test_loader = tf_data.build_loaders()
    
    if not test_loader: print("❌ Test set is empty."); return
    
    criterion = nn.MSELoss()
    trans_loss = 0.0
    tcn_loss = 0.0
    
    print(f"Evaluating on {len(test_loader)} test batches...")
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Transformer
            pred_t = transformer(batch_x)
            trans_loss += criterion(pred_t, batch_y).item()
            
            # TCN
            pred_c = tcn(batch_x)
            tcn_loss += criterion(pred_c, batch_y).item()
            
    avg_trans = trans_loss / len(test_loader)
    avg_tcn = tcn_loss / len(test_loader)
    
    print("\n" + "="*30)
    print("   FINAL TEST RESULTS   ")
    print("="*30)
    print(f"Transformer MSE:  {avg_trans:.6f}")
    print(f"TCN MSE:          {avg_tcn:.6f}")
    print("-" * 30)
    
    diff = avg_trans - avg_tcn
    if diff > 0:
        print(f"🏆 WINNER: TCN (Better by {diff:.6f})")
    else:
        print(f"🏆 WINNER: Transformer (Better by {abs(diff):.6f})")

if __name__ == "__main__":
    compare()
