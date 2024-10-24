from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MLP
import os

# Set CUDA_VISIBLE_DEVICES to GPU 1 only
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

# 调用函数并打印结果

MLP_input_size=39
MLP_hidden_size=200
MLP_output_size=1

device = torch.device("cuda")
model = MLP(input_size=MLP_input_size, hidden_size=MLP_hidden_size, output_size=MLP_output_size).to(device)
model_path="./checkpoint/model_checkpoint_9.pth"
model.load_state_dict(torch.load(model_path))
print(f"Total parameters in model: {count_parameters(model)}")
model.eval()
app = FastAPI()

class Item(BaseModel):
    inputs: list  # Expect a list of inputs

@app.post('/predict/')
async def predict(item: Item):
    inputs = torch.tensor(item.inputs).reshape((1,-1)).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    possibility=outputs.cpu().numpy().item()
    # if(possibility>0.5):
    #     label=1
    # else:
    #     label=0
    return {"prediction": possibility}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9102)
