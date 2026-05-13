import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Import BOTH architectures
from model import CNNHybridQNN, PureClassicalCNN

def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Shuffle ensures we get a random image each time we run the script
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True) 
    return next(iter(test_loader))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Comparative XAI on: {device}")
    
    # 1. Load the Quantum Model
    q_model = CNNHybridQNN().to(device)
    q_model.load_state_dict(torch.load("hybrid_qnn_mnist.pth", map_location=device))
    q_model.eval()

    # 2. Load the Classical Twin
    c_model = PureClassicalCNN().to(device)
    c_model.load_state_dict(torch.load("classical_cnn_mnist.pth", map_location=device))
    c_model.eval()

    # 3. Get ONE specific image to test on both
    images, labels = load_test_data()
    images, labels = images.to(device), labels.to(device)

    target_image = images[0].unsqueeze(0) # Shape: [1, 1, 28, 28]
    target_label = labels[0].item()
    
    # Quick sanity check on predictions
    with torch.no_grad():
        q_pred = q_model(target_image).argmax(dim=1).item()
        c_pred = c_model(target_image).argmax(dim=1).item()
    
    print(f"True Label: {target_label} | Quantum Pred: {q_pred} | Classical Pred: {c_pred}")

    # 4. Calculate Integrated Gradients for BOTH
    print("Calculating Quantum Gradients...")
    ig_q = IntegratedGradients(q_model)
    attr_q = ig_q.attribute(target_image, target=target_label, n_steps=10)

    print("Calculating Classical Gradients...")
    ig_c = IntegratedGradients(c_model)
    attr_c = ig_c.attribute(target_image, target=target_label, n_steps=10)

    # Convert to numpy for plotting
    original_img_np = target_image.squeeze().cpu().detach().numpy()
    attr_q_np = attr_q.squeeze().cpu().detach().numpy()
    attr_c_np = attr_c.squeeze().cpu().detach().numpy()

    # --- ZERO-GRADIENT SAFETY CHECKS ---
    if np.max(np.abs(attr_q_np)) == 0.0:
        print("[!] Quantum model returned zero gradients.")
        attr_q_np += 1e-12 
        
    if np.max(np.abs(attr_c_np)) == 0.0:
        print("[!] WARNING: Classical model went 'blind' (Dead ReLUs).")
        attr_c_np += 1e-12 
    # -----------------------------------

    # 5. Build the 3-Panel Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Original Image
    axes[0].imshow(original_img_np, cmap='gray')
    axes[0].set_title(f"Original Image\nTrue Label: {target_label}")
    axes[0].axis('off')
    
    # Panel 2: Quantum Heatmap
    viz.visualize_image_attr(
        np.expand_dims(attr_q_np, axis=2), 
        np.expand_dims(original_img_np, axis=2), 
        method="blended_heat_map", 
        sign="all", 
        show_colorbar=False, 
        title=f"Quantum Map (Pred: {q_pred})",
        plt_fig_axis=(fig, axes[1]),
        use_pyplot=False
    )

    # Panel 3: Classical Heatmap
    viz.visualize_image_attr(
        np.expand_dims(attr_c_np, axis=2), 
        np.expand_dims(original_img_np, axis=2), 
        method="blended_heat_map", 
        sign="all", 
        show_colorbar=True, 
        title=f"Classical Map (Pred: {c_pred})",
        plt_fig_axis=(fig, axes[2]),
        use_pyplot=False
    )
    
    plt.tight_layout()
    plt.savefig("benchmark_comparison.png", dpi=300)
    print("Saved side-by-side comparison as 'benchmark_comparison.png'!")