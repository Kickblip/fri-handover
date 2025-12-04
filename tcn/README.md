# Usage

# Train TCN
python -m tcn.train

# Train Transformer (existing)
python -m team_transformer.train

# Compare both (Computes MSE on test set)
python -m compare_models

# use Infer to visualize
python -m team_transformer.infer 40_video --video
python -m tcn.infer 40_video --video