# Usage

# Train TCN
python -m tcn.train

# Train Transformer (existing)
python -m team_transformer.train (on lab computer use different command)

# Compare both (Computes MSE on test set)
python -m compare_models

# use Infer to visualize
python -m team_transformer.infer 40_video --video (on lab computer use different command)
python -m tcn.infer 40_video --video

# Compare both (more metrics)
python compare_metrics.py