import yaml
import torch
import logging
from models import BPRMF, ItemKNN, DeepFM, LightGCN
from data.data_loader import DataManager, MovieLensDataset
from utils.evaluation import evaluate_model
from utils.logger import setup_logger
from torch.utils.data import DataLoader

logger = None

def train_model(model, train_data, val_data, config):
    """Train a recommender model and perform validation"""
    global logger
    
    device = torch.device(config['training']['device'])
    
    # Set up training parameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Set up early stopping
    best_val_score = float('-inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        # Create data loader with all required parameters
        train_dataset = MovieLensDataset(
            train_data, 
            config['data']['num_items'],
            config['data']['num_users'],
            negative_sample_size=4
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers']
        )
        
        # Training epoch
        for batch in train_loader:
            # Move batch to device, only for tensor objects
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            loss = model.calculate_loss(batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        val_metrics = evaluate_model(model, val_data, config)
        val_score = val_metrics[f"ndcg@{config['evaluation']['k_values'][-1]}"]
        
        # Logging
        logger.info(
            f"Epoch {epoch}: train_loss = {total_loss/len(train_loader):.4f}, "
            f"val_ndcg = {val_score:.4f}"
        )
        
        # Early stopping check
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

def main():
    global logger
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger()
    
    # Set device
    device = torch.device(config['training']['device'])
    
    # Load data
    data_manager = DataManager(config)
    data_splits = data_manager.load_data()
    
    # Initialize models
    models = {
        'BPRMF': BPRMF(config),
        'ItemKNN': ItemKNN(config),
        'DeepFM': DeepFM(config),
        'LightGCN': LightGCN(config)
    }
    
    # Train and evaluate each model
    results = {}
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model = model.to(device)
        
        # Train model
        train_model(model, data_splits['train'], data_splits['val'], config)
        
        # Evaluate model
        metrics = evaluate_model(model, data_splits['test'], config)
        results[model_name] = metrics
        
        logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()