"""
Block-wise XGB ensemble for handling large feature sets.
"""
import pandas as pd
from typing import List, Tuple
import logging
from .xgb_drivers import generate_xgb_specs, fold_train_predict

logger = logging.getLogger(__name__)

def create_feature_blocks(feature_names: List[str], block_size: int = 100) -> List[List[str]]:
    """Split features into blocks for specialized ensembles."""
    blocks = [feature_names[i:i + block_size] for i in range(0, len(feature_names), block_size)]
    logger.info(f"Created {len(blocks)} feature blocks (size ~{block_size})")
    return blocks

def train_block_ensemble(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, 
                        feature_block: List[str], n_models: int = 10, seed: int = 42) -> Tuple[List[pd.Series], List[pd.Series]]:
    """Train XGB ensemble on a specific feature block."""
    available_features = [f for f in feature_block if f in X_tr.columns]
    if not available_features:
        logger.warning("No valid features in block, returning zeros")
        zero_train = [pd.Series(0.0, index=X_tr.index, name=f"block_model_{i}") for i in range(n_models)]
        zero_test = [pd.Series(0.0, index=X_te.index, name=f"block_model_{i}") for i in range(n_models)]
        return zero_train, zero_test
    
    X_tr_block = X_tr[available_features]
    X_te_block = X_te[available_features]
    
    logger.info(f"Training block: {len(available_features)} features, {n_models} models")
    
    specs = generate_xgb_specs(n_models=n_models, seed=seed)
    return fold_train_predict(X_tr_block, y_tr, X_te_block, specs)

def train_all_blocks(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, 
                    feature_blocks: List[List[str]], models_per_block: int = 10, 
                    base_seed: int = 42) -> Tuple[List[List[pd.Series]], List[List[pd.Series]]]:
    """Train XGB ensembles on all feature blocks."""
    all_train_preds, all_test_preds = [], []
    
    for i, block in enumerate(feature_blocks):
        logger.info(f"Training block {i+1}/{len(feature_blocks)}")
        block_seed = base_seed + i * 1000
        train_preds, test_preds = train_block_ensemble(
            X_tr, y_tr, X_te, block, n_models=models_per_block, seed=block_seed
        )
        all_train_preds.append(train_preds)
        all_test_preds.append(test_preds)
    
    return all_train_preds, all_test_preds

def flatten_block_predictions(block_predictions: List[List[pd.Series]]) -> List[pd.Series]:
    """Flatten block predictions into single list for signal generation."""
    return [pred for block_preds in block_predictions for pred in block_preds]