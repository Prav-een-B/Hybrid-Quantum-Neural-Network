import torch
import torch.nn as nn
import pennylane as qml

# 1. Define the Quantum Device (4 qubits)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 2. Define the Quantum Circuit (QNode)
@qml.qnode(dev, interface="torch")
def quantum_net(inputs, weights):
    # Embed classical data into quantum state
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Apply parameterized quantum gates
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # Measure expectation value of Pauli-Z on all qubits
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# 3. The Upgraded CNN-Quantum Hybrid Architecture
class CNNHybridQNN(nn.Module):
    def __init__(self):
        super(CNNHybridQNN, self).__init__()
        
        # --- NEW: Convolutional Feature Extractor ---
        # Preserves spatial relationships while compressing the image
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1), # Output: [batch, 4, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                           # Output: [batch, 4, 14, 14]
            nn.Conv2d(4, 8, kernel_size=3, padding=1), # Output: [batch, 8, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                           # Output: [batch, 8, 7, 7]
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, n_qubits)             # Bottleneck: Project down to exactly 4 features
        )
        
        # --- Quantum Layer Setup ---
        weight_shapes = {"weights": (2, n_qubits)}     # 2 entangling layers
        self.qlayer = qml.qnn.TorchLayer(quantum_net, weight_shapes)
        
        # --- Classical Output Layer ---
        self.fc_out = nn.Linear(n_qubits, 10)          # Map 4 quantum outputs to 10 MNIST classes

    def forward(self, x):
        # 1. Extract spatial features using the CNN
        x = self.conv_encoder(x)
        
        # 2. Normalize features to [0, pi] for Quantum Angle Embedding
        x = torch.sigmoid(x) * torch.pi 
        
        # 3. Process through the Quantum Circuit
        x = self.qlayer(x)
        
        # 4. Final classification
        x = self.fc_out(x)
        return x

# Add this inside model.py

class PureClassicalCNN(nn.Module):
    def __init__(self):
        super(PureClassicalCNN, self).__init__()
        
        # EXACT same encoder as the quantum model
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 4) # The same 4-feature bottleneck
        )
        
        # Classical Output Layer (replacing the quantum layer)
        self.fc_out = nn.Linear(4, 10) 

    def forward(self, x):
        x = self.conv_encoder(x)
        
        # Standard classical activation instead of the quantum Angle Embedding
        x = torch.relu(x) 
        
        x = self.fc_out(x)
        return x