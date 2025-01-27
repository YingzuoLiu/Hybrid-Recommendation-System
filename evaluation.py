import torch
import numpy as np
from tqdm import tqdm
from .metrics import calculate_metrics

def evaluate_model(model, test_data, config):
    """Evaluate model performance on test data
    
    Args:
        model: The recommender model to evaluate
        test_data: Test data dictionary
        config: Configuration dictionary
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    # Initialize metrics dictionary
    metrics_dict = {}
    for metric in config['evaluation']['metrics']:
        for k in config['evaluation']['k_values']:
            metrics_dict[f"{metric}@{k}"] = []
    
    with torch.no_grad():
        for user in tqdm(test_data.keys()):
            # Get user's test items
            test_items = test_data[user]
            
            # Get recommendations
            user_tensor = torch.tensor([user]).to(config['training']['device'])
            item_tensors = torch.arange(config['data']['num_items']).to(config['training']['device'])
            
            predictions = model.predict(user_tensor, item_tensors)
            predictions = predictions.cpu().numpy()
            
            # Calculate metrics
            for metric in config['evaluation']['metrics']:
                for k in config['evaluation']['k_values']:
                    metric_key = f"{metric}@{k}"
                    score = calculate_metrics(predictions, test_items, metric, k)
                    metrics_dict[metric_key].append(score)
    
    # Average metrics
    final_metrics = {}
    for metric_key, scores in metrics_dict.items():
        final_metrics[metric_key] = np.mean(scores)
    
    return final_metrics