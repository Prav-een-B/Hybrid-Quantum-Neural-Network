# explain.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from captum.attr import GradientShap, IntegratedGradients
from captum.attr import visualization as viz

# Import your architecture
from model import CNNHybridQNN

def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Batch size 5 just to grab a few samples for visualization
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True) 
    return next(iter(test_loader))

if __name__ == "__main__":
    # 1. Setup Device and Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running XAI on: {device}")
    
    model = CNNHybridQNN().to(device)
    model.load_state_dict(torch.load("hybrid_qnn_mnist.pth", map_location=device))
    model.eval() # CRITICAL: Put model in evaluation mode

    # 2. Get Sample Data
    images, labels = load_test_data()
    images, labels = images.to(device), labels.to(device)

    # We will pick the first image in our random batch to explain
    target_image = images[0].unsqueeze(0) # Shape: [1, 1, 28, 28]
    target_label = labels[0].item()
    
    # Run a quick prediction to see what the model thinks it is
    with torch.no_grad():
        pred = model(target_image)
        pred_label = pred.argmax(dim=1).item()
    
    print(f"True Label: {target_label} | Model Predicted: {pred_label}")

    # 3. Initialize Explainability Methods (GradientSHAP & Integrated Gradients)
    # We use a baseline of zeros (a black image) to compare our actual image against
    baseline = torch.zeros_like(target_image).to(device)
    
    print("Calculating Integrated Gradients (this involves quantum simulation, may take a minute...)")
    ig = IntegratedGradients(model)
    
    # Calculate attributions for the predicted class
    attributions_ig = ig.attribute(target_image, target=pred_label, n_steps=10)

    # 4. Visualization & Plotting
    # Convert tensors to numpy arrays for Matplotlib (shape: 28x28)
    original_img_np = target_image.squeeze().cpu().detach().numpy()
    attr_ig_np = attributions_ig.squeeze().cpu().detach().numpy()

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot Original Image
    axes[0].imshow(original_img_np, cmap='gray')
    axes[0].set_title(f"Original Image\nTrue: {target_label} | Pred: {pred_label}")
    axes[0].axis('off')
    
    # Plot Saliency Map overlay (Heatmap)
    # Red pixels = pushed the model toward this prediction
    # Blue pixels = pushed the model away from this prediction
    viz.visualize_image_attr(
        np.expand_dims(attr_ig_np, axis=2), 
        np.expand_dims(original_img_np, axis=2), 
        method="blended_heat_map", 
        sign="all", 
        show_colorbar=True, 
        title="Quantum Interpretability Map",
        plt_fig_axis=(fig, axes[1]),
        use_pyplot=False
    )
    
    plt.tight_layout()
    plt.savefig("quantum_explain_output.png", dpi=300)
    print("Saved explainability map as 'quantum_explain_output.png'!")