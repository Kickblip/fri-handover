# Handover Future Frame Prediction Model

This model predicts future frames of the receiving hand during handover events.

## Training

Train the model:
```
python -m model.train
```

## Inference

Run inference and generate predictions:
```
python -m model.infer 1_w_b
```

Generate video visualization of predicted future frames:
```
python -m model.infer 1_w_b --video
```

## Description

The model predicts 5 frames into the future for the receiving hand (hand_1) based on the input sequence of both hands' features. It uses a Transformer encoder-decoder architecture:
- **Input**: Sequences of both hands' Rodrigues rotation vectors and optional vertices
- **Output**: Predicted world coordinates of the receiving hand for the next 5 frames

## Outputs

- **Checkpoints**: Saved to `dataset/model_output/checkpoints/handover_transformer.pt`
- **Predictions**: CSV files saved to `dataset/model_output/predictions/<stem>_future_predictions.csv`
- **Videos**: Video visualizations saved to `dataset/model_output/videos/<stem>_predicted_future.mp4`
