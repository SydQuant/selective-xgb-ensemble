"""
Block-wise XGB ensemble strategy for handling large feature sets.
Instead of single massive models, create specialized block ensembles.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from .xgb_drivers import generate_xgb_specs, fold_train_predict

logger = logging.getLogger(__name__)

def create_feature_blocks(feature_names: List[str], block_size: int = 100) -> List[List[str]]:
    """Split features into blocks for specialized ensembles."""
    blocks = []
    for i in range(0, len(feature_names), block_size):
        block = feature_names[i:i + block_size]
        blocks.append(block)
    
    logger.info(f"Created {len(blocks)} feature blocks (size ~{block_size})")
    return blocks

def train_block_ensemble(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, 
                        feature_block: List[str], n_models: int = 10, seed: int = 42) -> Tuple[List[pd.Series], List[pd.Series]]:
    """Train XGB ensemble on a specific feature block."""
    
    # Select features for this block
    available_features = [f for f in feature_block if f in X_tr.columns]
    if len(available_features) == 0:
        logger.warning(f"No valid features in block, returning zeros")
        zero_train = [pd.Series(0.0, index=X_tr.index, name=f"block_model_{i}") for i in range(n_models)]
        zero_test = [pd.Series(0.0, index=X_te.index, name=f"block_model_{i}") for i in range(n_models)]
        return zero_train, zero_test
    
    X_tr_block = X_tr[available_features]
    X_te_block = X_te[available_features]
    
    logger.info(f"Training block ensemble: {len(available_features)} features, {n_models} models")
    
    # Generate specs for this block
    specs = generate_xgb_specs(n_models=n_models, seed=seed)
    
    # Train models on block
    train_preds, test_preds = fold_train_predict(X_tr_block, y_tr, X_te_block, specs)
    
    return train_preds, test_preds

def train_all_blocks(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, 
                    feature_blocks: List[List[str]], models_per_block: int = 10, 
                    base_seed: int = 42) -> Tuple[List[List[pd.Series]], List[List[pd.Series]]]:
    """Train XGB ensembles on all feature blocks."""
    
    all_train_preds = []
    all_test_preds = []
    
    for i, block in enumerate(feature_blocks):
        logger.info(f"Training block {i+1}/{len(feature_blocks)}")
        
        block_seed = base_seed + i * 1000  # Ensure different seeds per block
        train_preds, test_preds = train_block_ensemble(
            X_tr, y_tr, X_te, block, n_models=models_per_block, seed=block_seed
        )
        
        all_train_preds.append(train_preds)
        all_test_preds.append(test_preds)
    
    return all_train_preds, all_test_preds

def flatten_block_predictions(block_predictions: List[List[pd.Series]]) -> List[pd.Series]:
    """Flatten block predictions into single list for signal generation."""
    flattened = []
    for block_preds in block_predictions:
        flattened.extend(block_preds)
    return flattened