"""
Block-wise feature selection for large feature sets.
"""
import numpy as np
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)

def block_wise_feature_selection(X: pd.DataFrame, y: pd.Series, 
                                block_size: int = 100,
                                features_per_block: int = None,  # Legacy compatibility
                                max_total_features: int = 50,
                                corr_threshold: float = 0.7) -> List[str]:
    """Simplified correlation-based feature selection with threshold filtering and hard limit."""
    # Handle legacy parameter
    if features_per_block is not None:
        block_size = features_per_block
        
    feature_names = X.columns.tolist()
    
    if len(feature_names) <= max_total_features:
        return feature_names
    
    # Step 1: Process blocks - get all uncorrelated features within each block
    blocks = [feature_names[i:i + block_size] for i in range(0, len(feature_names), block_size)]
    all_candidates = []
    
    for block in blocks:
        X_block = X[block]
        
        # Remove zero variance features
        valid_features = [f for f in block if X_block[f].std() > 1e-10]
        if not valid_features:
            continue
            
        # Correlation-based filtering within block (no hard limit)
        block_selected = []
        for feat in valid_features:
            keep_feature = True
            for selected_feat in block_selected:
                try:
                    feat_corr = X_block[feat].corr(X_block[selected_feat])
                    if pd.notna(feat_corr) and abs(feat_corr) > corr_threshold:
                        keep_feature = False
                        break
                except:
                    pass
            
            if keep_feature:
                block_selected.append(feat)
        
        all_candidates.extend(block_selected)
    
    # Step 2: Inter-block deduplication with correlation filtering
    deduplicated = []
    for feat in all_candidates:
        keep_feature = True
        for selected_feat in deduplicated:
            try:
                feat_corr = X[feat].corr(X[selected_feat])
                if pd.notna(feat_corr) and abs(feat_corr) > corr_threshold:
                    keep_feature = False
                    break
            except:
                pass
        
        if keep_feature:
            deduplicated.append(feat)
    
    # Step 3: Hard limit - take first N features if more than max_total_features
    if len(deduplicated) > max_total_features:
        final_selected = deduplicated[:max_total_features]
    else:
        final_selected = deduplicated
    
    logger.info(f"Selected {len(final_selected)} features (threshold: {corr_threshold})")
    return final_selected

def apply_feature_selection(X: pd.DataFrame, y: pd.Series, method: str = 'block_wise', **kwargs) -> pd.DataFrame:
    """Apply feature selection method and return reduced dataset."""
    
    if method == 'block_wise':
        selected_features = block_wise_feature_selection(X, y, **kwargs)
        return X[selected_features]
    elif method == 'none':
        return X
    else:
        raise ValueError(f"Unknown feature selection method: {method}")