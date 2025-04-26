import torch

# Learning rate scheduling function
def get_scheduler(optimizer, schedule_type='cosine'):
    if schedule_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif schedule_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

# Utility to save results in JSON format
import json

def save_results(results, file_name):
    # Convert any tensors to lists before saving to JSON
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert tensor to list
        if isinstance(obj, dict):
            return {key: convert_to_serializable(val) for key, val in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(val) for val in obj]
        return obj  # Return the object as is if it's not a tensor or a dict/list

    # Convert the entire results dictionary to a serializable format
    serializable_results = convert_to_serializable(results)
    
    # Save the converted results to JSON
    with open(file_name, "w") as f:
        json.dump(serializable_results, f, indent=4)
