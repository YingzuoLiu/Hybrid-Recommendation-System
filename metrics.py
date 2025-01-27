import numpy as np
from sklearn.metrics import ndcg_score

def calculate_metrics(predictions, true_items, metric_name, k):
    """
    Calculate recommendation metrics
    
    Args:
        predictions: numpy array of predicted scores for all items
        true_items: numpy array of actual items that user has interacted with
        metric_name: string indicating which metric to calculate
        k: cut-off value for top-K metrics
    
    Returns:
        float: calculated metric value
    """
    # Get top-k item indices
    top_k_items = np.argsort(-predictions)[:k]
    
    if metric_name == "recall":
        return calculate_recall(top_k_items, true_items)
    elif metric_name == "precision":
        return calculate_precision(top_k_items, true_items)
    elif metric_name == "ndcg":
        return calculate_ndcg(predictions, true_items, k)
    elif metric_name == "hit_rate":
        return calculate_hit_rate(top_k_items, true_items)
    elif metric_name == "map":
        return calculate_map(top_k_items, true_items)
    else:
        raise ValueError(f"Metric {metric_name} not implemented")

def calculate_recall(recommended_items, true_items):
    """Calculate Recall@K"""
    intersection = np.intersect1d(recommended_items, true_items)
    return len(intersection) / len(true_items) if len(true_items) > 0 else 0.0

def calculate_precision(recommended_items, true_items):
    """Calculate Precision@K"""
    intersection = np.intersect1d(recommended_items, true_items)
    return len(intersection) / len(recommended_items)

def calculate_ndcg(predictions, true_items, k):
    """Calculate NDCG@K"""
    # Create binary relevance array
    relevance = np.zeros_like(predictions)
    relevance[true_items] = 1
    
    # Reshape for sklearn's ndcg_score
    true_relevance = relevance.reshape(1, -1)
    predictions = predictions.reshape(1, -1)
    
    return ndcg_score(true_relevance, predictions, k=k)

def calculate_hit_rate(recommended_items, true_items):
    """Calculate Hit Rate@K"""
    intersection = np.intersect1d(recommended_items, true_items)
    return 1.0 if len(intersection) > 0 else 0.0

def calculate_map(recommended_items, true_items):
    """Calculate Mean Average Precision@K"""
    hits = np.isin(recommended_items, true_items)
    if not hits.any():
        return 0.0
    
    precision_at_k = np.cumsum(hits) / np.arange(1, len(recommended_items) + 1)
    return np.sum(precision_at_k * hits) / min(len(true_items), len(recommended_items))