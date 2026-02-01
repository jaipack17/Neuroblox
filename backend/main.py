import sys
import os
import subprocess
import uuid
import json
import logging
import importlib.util
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, model_validator
from typing import List, Optional, Literal, Dict

# --- LOGGING SETUP ---
logger = logging.getLogger("Backend")
logger.setLevel(logging.INFO)

# Console Handler
c_handler = logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(c_handler)

# File Handler
f_handler = logging.FileHandler("server.log")
f_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(f_handler)

# --- IMPORTS ---
try:
    from backend.generator import create_pytorch_script
except ImportError:
    # Fallback if running directly inside backend folder
    from generator import create_pytorch_script

# --- CONFIG ---
GENERATED_DIR = "generated_models"
training_state: Dict[str, dict] = {}

# --- PYDANTIC MODELS ---

class DatasetConfig(BaseModel):
    type: Literal["csv", "image"]
    path: str
    batch_size: int = 16 

class Block(BaseModel):
    type: Literal["dense", "relu", "sigmoid", "dropout", "softmax"]
    size: Optional[int] = None
    prob: Optional[float] = None

    @model_validator(mode='after')
    def check_reqs(self):
        if self.type == 'dense' and self.size is None:
            raise ValueError('Dense layers require a "size" field')
        if self.type == 'dropout' and self.prob is None:
            raise ValueError('Dropout layers require a "prob" field')
        return self

class TrainRequest(BaseModel):
    epochs: int
    test_split: float = 0.2  # Default 20% test split
    dataset: DatasetConfig
    blocks: List[Block]

class PredictRequest(BaseModel):
    job_id: str
    input_data: List[float]  # e.g., [5.1, 3.5, 1.4, 0.2]

# --- BACKGROUND RUNNER ---

def execute_training(script_path: str, job_id: str):
    """
    Executes the training script and captures live JSON logs.
    """
    logger.info(f"Job [{job_id}]: Starting training script {script_path}")
    training_state[job_id] = {"status": "starting", "epoch": 0}

    try:
        # bufsize=1 enables line-buffered output for real-time updates
        with subprocess.Popen(
            [sys.executable, script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        ) as process:
            
            # Read stdout line by line
            for line in process.stdout:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        update = json.loads(line)
                        training_state[job_id].update(update)
                        
                        # Log key events to server log
                        if update.get("status") == "complete":
                            logger.info(f"Job [{job_id}] Complete. Final Accuracy: {update.get('final_accuracy')}%")
                    except json.JSONDecodeError:
                        pass # Ignore non-JSON prints

            # Check for crashes
            stderr = process.stderr.read()
            if stderr:
                logger.error(f"Job [{job_id}] Script Error: {stderr}")
                training_state[job_id]["status"] = "failed"
                training_state[job_id]["error"] = stderr[:500] # Limit error length

    except Exception as e:
        logger.critical(f"Job [{job_id}] Execution failed: {e}")
        training_state[job_id]["status"] = "failed"
        training_state[job_id]["error"] = str(e)
    
    # Final cleanup
    if training_state[job_id].get("status") != "failed":
        if training_state[job_id].get("status") != "complete":
             training_state[job_id]["status"] = "complete"

# --- HELPER: DYNAMIC IMPORT ---
def load_module_from_path(path: str, module_name: str):
    """Loads a python file as a module dynamically."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- API ENDPOINTS ---

app = FastAPI()

@app.post("/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    payload = request.model_dump()
    
    logger.info(f"New Training Request: ID={job_id}, Split={request.test_split}, Epochs={request.epochs}")

    try:
        script_path = create_pytorch_script(payload, job_id)
        background_tasks.add_task(execute_training, script_path, job_id)
        
        return {
            "message": "Training Initiated",
            "job_id": job_id,
            "monitor_url": f"/status/{job_id}"
        }
    except Exception as e:
        logger.error(f"Failed to generate script: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    status = training_state.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return status

@app.get("/models")
async def list_models():
    """Lists all generated models available for inference."""
    if not os.path.exists(GENERATED_DIR):
        return []
        
    models = []
    # Scan directory for model_*.py files
    for f in os.listdir(GENERATED_DIR):
        if f.startswith("model_") and f.endswith(".py"):
            jid = f.replace("model_", "").replace(".py", "")
            models.append({"job_id": jid, "filename": f})
            
    return models

@app.post("/predict")
async def run_inference(request: PredictRequest):
    """
    Loads the specific model script and calls its predict() function.
    """
    script_name = f"model_{request.job_id}.py"
    script_path = os.path.join(GENERATED_DIR, script_name)
    
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail="Model file not found. Has it been trained?")

    try:
        # Load the script dynamically
        model_module = load_module_from_path(script_path, f"mod_{request.job_id}")
        
        # Call predict(input_data)
        result = model_module.predict(request.input_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Inference Error on Job {request.job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Backend...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
