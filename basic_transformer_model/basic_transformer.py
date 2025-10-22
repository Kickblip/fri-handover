# This code includes a mock data generator so you can run and test the script immediately, but you 
# should replace the mock data with your actual loaded CSV data. The task is structured as a 
# binary classification (e.g., predicting if a handover is occurring in the next frame).

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================
# ==== CONFIGURATION ====
# ============================

# Model/Training Hyperparameters
SEQUENCE_LENGTH = 30    # Number of frames in one input sequence (e.g., 1 second at 30 FPS)
BATCH_SIZE = 64         # Number of sequences processed simultaneously
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
NUM_FEATURES = 138      # Must match the number of coordinate columns in your combined CSV (63*2 + 2 meta=128, 63*2=126, let's assume 126 coordinate features)

# Transformer Architecture
D_MODEL = 64            # The embedding dimension for the Transformer
N_HEAD = 4              # Number of attention heads
N_LAYERS = 2            # Number of encoder layers
DIM_FEEDFORWARD = 128   # Dimension of the feedforward network
NUM_CLASSES = 1         # Binary classification (0 or 1)
DROPOUT = 0.1

# Data Paths (Replace with your actual file path)
INPUT_CSV = Path("outputs/handover_features_simplified.csv") 

# ============================
# ==== DATA PREPARATION ====
# ============================

def load_and_preprocess_data(file_path: Path) -> pd.DataFrame:
    """
    Loads the combined Cartesian data and selects only the feature columns.
    NOTE: Replace this with your actual data loading logic.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Generating mock data for demonstration.")
        return generate_mock_data(2000, NUM_FEATURES) # 2000 total frames of mock data

    # Assumes your features are the coordinate columns with suffixes _0 and _1
    # Filter out 'time_sec', 'frame_index', and 'hand_label/score' 
    feature_cols = [col for col in df.columns if col not in ['time_sec', 'frame_index', 'hand_label_0', 'hand_label_1', 'hand_score_0', 'hand_score_1']]
    
    # Fill NaN values (for frames with only one hand) with 0 or mean/median
    # Using 0 for simplicity, but consider more robust imputation in production
    data = df[feature_cols].fillna(0)
    
    # Verify feature count
    if data.shape[1] != NUM_FEATURES:
        print(f"WARNING: Feature count mismatch. Expected {NUM_FEATURES}, got {data.shape[1]}. Adjust NUM_FEATURES.")

    return data

def generate_mock_data(num_frames: int, num_features: int) -> pd.DataFrame:
    """Generates random data for testing the model architecture."""
    # Mock Features (X): Random coordinates
    X_data = np.random.randn(num_frames, num_features).astype(np.float32)
    col_names = [f'feature_{i}' for i in range(num_features)]
    df_X = pd.DataFrame(X_data, columns=col_names)
    
    # Mock Target (Y): Random binary labels (0 or 1) for the handover event
    Y_data = np.random.randint(0, 2, size=(num_frames, 1)).astype(np.float32)
    df_Y = pd.DataFrame(Y_data, columns=['target'])

    return pd.concat([df_X, df_Y], axis=1)

def create_sequences(data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts the flat DataFrame into 3D tensors using a sliding window.
    X shape: (num_sequences, sequence_length, num_features)
    Y shape: (num_sequences, num_classes)
    """
    data_array = data.values
    features = data_array[:, :-1] # All columns except the last (target)
    targets = data_array[:, -1]    # The last column (target)
    
    X, Y = [], []
    
    # Simple sliding window
    for i in range(len(features) - SEQUENCE_LENGTH):
        # Input sequence (SEQUENCE_LENGTH frames)
        X.append(features[i:i + SEQUENCE_LENGTH])
        # Target for the sequence is the event at the LAST frame of the sequence
        Y.append(targets[i + SEQUENCE_LENGTH - 1])
        
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1) # [N] -> [N, 1]

    print(f"Created {X_tensor.shape[0]} sequences.")
    print(f"X shape: {X_tensor.shape}, Y shape: {Y_tensor.shape}")
    
    return X_tensor, Y_tensor

# ============================
# ==== TRANSFORMER MODEL ====
# ============================

class HandoverTransformer(nn.Module):
    """
    A basic sequence classification model based on the Transformer Encoder.
    """
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, 
                 dim_feedforward: int, dropout: float, num_features: int, num_classes: int):
        super().__init__()
        
        # 1. Input Embedding: Project high-dim features to d_model
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 2. Positional Encoding (Crucial for Transformers - simplified here)
        # For simplicity, we use a learned positional embedding.
        self.pos_encoder = nn.Parameter(torch.randn(SEQUENCE_LENGTH, d_model))
        
        # 3. Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_encoder_layers
        )
        
        # 4. Classification Head
        # We classify based on the output of the last token (last frame)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
            nn.Sigmoid() # Use Sigmoid for binary classification
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (B, S, F) -> (Batch, Sequence, Features)

        # 1. Input Projection
        src = self.input_projection(src) # (B, S, F) -> (B, S, D_MODEL)
        
        # 2. Add Positional Encoding
        # Positional encoding is added to the feature embeddings
        src = src + self.pos_encoder.unsqueeze(0) # (B, S, D_MODEL)
        
        # 3. Transformer Encoder
        output = self.transformer_encoder(src) # (B, S, D_MODEL)
        
        # 4. Classification
        # We take the output corresponding to the LAST frame/token (index -1)
        last_frame_output = output[:, -1, :] # (B, D_MODEL)
        
        # Pass through the classifier head
        output_prob = self.classifier(last_frame_output) # (B, NUM_CLASSES)
        
        return output_prob

# ============================
# ==== TRAINING FUNCTION ====
# ============================

def train_model():
    """Main function to run the data loading, model setup, and training."""
    
    # 1. Data Preparation
    full_data = load_and_preprocess_data(INPUT_CSV)
    if full_data.empty or full_data.shape[0] < SEQUENCE_LENGTH:
        print("Error: Not enough data to create sequences. Exiting.")
        return

    X, Y = create_sequences(full_data)
    
    # Split data (using all data for simplicity in this template)
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model, Loss, and Optimizer Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HandoverTransformer(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_features=X.shape[2], # Use the actual feature count from data
        num_classes=NUM_CLASSES
    ).to(device)

    # For binary classification, use Binary Cross Entropy Loss
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    print("\nTraining complete.")
    # Optional: Save the trained model
    # torch.save(model.state_dict(), "handover_transformer_model.pth")
    # print("Model saved to handover_transformer_model.pth")


if __name__ == "__main__":
    # NOTE: Before running, ensure you have PyTorch installed: 
    # pip install torch pandas numpy
    
    # Adjust NUM_FEATURES based on the final column count in your combined CSV!
    train_model()
