# Hybrid Quantum Neural Network

This repository explores explainability techniques on both classical and hybrid quantum–classical neural networks for the MNIST digit classification task. It combines a small CNN feature extractor with a PennyLane quantum layer, then uses Captum to generate saliency maps that interpret the model’s predictions.

## What’s inside
- `model.py`: Hybrid CNN + quantum circuit (`CNNHybridQNN`) and a baseline classical CNN (`PureClassicalCNN`).
- `train.py`: Trains the model on MNIST and saves weights to `hybrid_qnn_mnist.pth`.
- `explain.py`: Runs Integrated Gradients to create a heatmap and saves `quantum_explain_output.png`.
- `data/`: MNIST dataset cache.

## Quick start
1. Install dependencies:
   - `pip install -r requirements.txt`
   - For explainability: `pip install captum matplotlib numpy`
2. Train the hybrid model:
   - `python train.py`
3. Generate an explainability map:
   - `python explain.py`

## Notes
- The training script defaults to the hybrid model; switch to the classical baseline by instantiating `PureClassicalCNN` in `train.py`.
- MNIST will download automatically to `data/` on first run.
- This is still under development.
