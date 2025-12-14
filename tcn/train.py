import sys
import os
from pathlib import Path

# Add the current directory to path so we can import local config/data
# Use 'insert(0, ...)' to FORCE looking in the local folder FIRST
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Add the root directory to path so we can find shared utils if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import tcn_config directly
import tcn_config
import data
from model import TemporalConvNet
import utils 

def train(stems_to_use=None):
    utils.set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load from local TCN data loader
    train_ld, val_ld, test_ld = data.build_loaders(stems_to_use)
    
    if stems_to_use is None:
        stems = data.list_stems()
    else:
        stems = stems_to_use

    # Get dims from first stem
    in_dim = data.load_features(stems[0])[0].shape[1]
    out_dim = data.load_receiving_hand_world(stems[0])[0].shape[1]

    print(f"TCN Config: {tcn_config.TCN_HIDDEN_CHANNELS}, Kernel: {tcn_config.TCN_KERNEL_SIZE}")

    model = TemporalConvNet(
        in_dim=in_dim,
        num_channels=tcn_config.TCN_HIDDEN_CHANNELS,
        out_dim=out_dim,
        future_frames=tcn_config.FUTURE_FRAMES,
        kernel_size=tcn_config.TCN_KERNEL_SIZE,
        dropout=tcn_config.TCN_DROPOUT
    ).to(device)
    
    criterion = nn.MSELoss()
    opt = AdamW(model.parameters(), lr=tcn_config.LR)
    
    # Removed verbose=True to support newer PyTorch versions
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience_counter = 0  # Initialize patience counter
    
    ckpt_path = tcn_config.CKPT_DIR / "tcn_handover.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, tcn_config.MAX_EPOCHS + 1):
        model.train()
        total_loss = 0
        count = 0
        
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
        
        train_loss = total_loss / count

        # Validation
        model.eval()
        val_loss = 0
        if val_ld:
            val_total = 0
            val_count = 0
            with torch.no_grad():
                for x, y in val_ld:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_total += loss.item() * x.size(0)
                    val_count += x.size(0)
            val_loss = val_total / val_count
            sched.step(val_loss)
            
            print(f"Epoch {epoch} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
            # --- EARLY STOPPING LOGIC ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter when we improve
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'in_dim': in_dim,
                        'num_channels': tcn_config.TCN_HIDDEN_CHANNELS,
                        'out_dim': out_dim,
                        'kernel_size': tcn_config.TCN_KERNEL_SIZE
                    }
                }, ckpt_path)
                print(f"  -> Saved best model: {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= tcn_config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
            # -----------------------------

if __name__ == "__main__":
    train()