import os
import json

# Define paths
OUTPUT_DIR = "generated_models"
SAVED_MODELS_DIR = "saved_models"

def ensure_dirs():
    """Creates necessary directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def create_pytorch_script(data: dict, job_id: str):
    """
    Generates a Python script for training and inference based on the JSON config.
    """
    ensure_dirs()
    
    # 1. Extract Configuration
    blocks = data.get("blocks", [])
    epochs = data.get("epochs", 10)
    test_split = data.get("test_split", 0.2)  # Default 20% split
    
    dataset_config = data.get("dataset", {})
    batch_size = dataset_config.get("batch_size", 32)
    csv_path = dataset_config.get("path", "data.csv")
    
    # Path where weights will be saved
    weights_path = os.path.join(SAVED_MODELS_DIR, f"{job_id}.pth")

    # --- PART 1: IMPORTS & CONFIG (Using f-string for variable injection) ---
    # We use .replace(os.sep, '/') to ensure paths work on Windows/Linux inside the string
    code_header = f"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import sys
import json
import os

# --- CONFIGURATION ---
CSV_PATH = "{csv_path}"
WEIGHTS_PATH = "{weights_path.replace(os.sep, '/')}"
EPOCHS = {epochs}
BATCH_SIZE = {batch_size}
TEST_SPLIT = {test_split}
"""

    # --- PART 2: STATIC FUNCTIONS (Standard String - No escaping needed) ---
    code_static_functions = """
def load_and_split_data(filepath):
    try:
        # Load Data (Assuming features then target in last column)
        df = pd.read_csv(filepath)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.longlong)
        
        full_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        
        # Calculate Split Sizes
        total_size = len(full_dataset)
        test_size = int(total_size * TEST_SPLIT)
        train_size = total_size - test_size
        
        # Perform Split (Seed ensures reproducibility)
        torch.manual_seed(42)
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        return train_dataset, test_dataset, X.shape[1]
    except Exception as e:
        print(json.dumps({"status": "failed", "error": f"Data Load Error: {str(e)}"}), flush=True)
        sys.exit(1)
"""

    # --- PART 3: MODEL DEFINITION (Dynamic Construction) ---
    code_model_def = """
class NeuralNetwork(nn.Module):
    def __init__(self, input_shape):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
"""
    # Logic to build layers
    current_input_dim = "input_shape" 
    
    for block in blocks:
        kind = block.get("type")
        
        if kind == "dense":
            out_dim = block.get("size")
            code_model_def += f"            nn.Linear({current_input_dim}, {out_dim}),\n"
            current_input_dim = out_dim 
            
        elif kind == "relu":
            code_model_def += "            nn.ReLU(),\n"
        elif kind == "sigmoid":
            code_model_def += "            nn.Sigmoid(),\n"
        elif kind == "softmax":
            code_model_def += "            nn.Softmax(dim=1),\n"
        elif kind == "dropout":
            prob = block.get("prob", 0.5)
            code_model_def += f"            nn.Dropout({prob}),\n"

    code_model_def += """        )

    def forward(self, x):
        return self.layers(x)
"""

    # --- PART 4: INFERENCE & TRAINER (Standard String) ---
    code_execution = """
def predict(input_data):
    '''
    Called by the backend to run inference.
    input_data: List of floats
    '''
    try:
        input_dim = len(input_data)
        model = NeuralNetwork(input_dim)
        
        if not os.path.exists(WEIGHTS_PATH):
            return {"error": "Model weights not found. Train the model first."}
            
        model.load_state_dict(torch.load(WEIGHTS_PATH))
        model.eval()
        
        tensor_in = torch.tensor([input_data], dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor_in)
            
        predicted_class = torch.argmax(output, dim=1).item()
        return {"prediction": predicted_class, "raw_output": output.tolist()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # 1. Notify Start
    print(json.dumps({"status": "starting"}), flush=True)

    try:
        # 2. Load Data
        train_dataset, test_dataset, input_dim = load_and_split_data(CSV_PATH)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # Process validation set in one go (or batches if large)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) or 1, shuffle=False)
        
        # 3. Setup Model
        model = NeuralNetwork(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        # 4. Training Loop
        for epoch in range(EPOCHS):
            # A. Train Step
            model.train()
            running_loss = 0.0
            for _, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            
            # B. Validation Step (Test Split)
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in test_loader:
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_accuracy = (100 * correct / total) if total > 0 else 0.0
            avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else 0.0

            # C. Log Update
            print(json.dumps({
                "status": "training",
                "epoch": epoch + 1,
                "total_epochs": EPOCHS,
                "loss": round(avg_train_loss, 4),
                "val_loss": round(avg_val_loss, 4),
                "val_accuracy": round(val_accuracy, 2)
            }), flush=True)

        # 5. Save Model
        torch.save(model.state_dict(), WEIGHTS_PATH)
        
        print(json.dumps({
            "status": "complete", 
            "final_loss": round(avg_train_loss, 4),
            "final_accuracy": round(val_accuracy, 2)
        }), flush=True)

    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e)}), flush=True)
        sys.exit(1)
"""

    # --- JOIN PARTS ---
    full_code = code_header + code_static_functions + code_model_def + code_execution

    # --- SAVE FILE ---
    filename = f"model_{job_id}.py"
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(file_path, "w") as f:
        f.write(full_code)
        
    return file_path
