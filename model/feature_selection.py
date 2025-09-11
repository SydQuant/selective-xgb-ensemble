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
    
    # Handle -1 (no limit) and early return for small datasets
    if max_total_features == -1 or len(feature_names) <= max_total_features:
        # If no limit or already within limit, apply correlation filtering only
        if max_total_features == -1:
            max_total_features = len(feature_names)  # Set to total available
        elif len(feature_names) <= max_total_features:
            return feature_names  # No selection needed
    
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
    
    # Step 3: Optimized diverse subset selection
    if max_total_features > 0 and len(deduplicated) > max_total_features:
        # Vectorized target correlation calculation
        X_subset = X[deduplicated]
        target_corrs = X_subset.corrwith(y).abs().fillna(0.0)
        
        # Sort features by target correlation (descending)
        feature_target_pairs = [(feat, target_corrs[feat]) for feat in deduplicated]
        feature_target_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Pre-compute correlation matrix for efficiency
        corr_matrix = X_subset.corr().abs()
        
        final_selected = []
        available_indices = set(range(len(feature_target_pairs)))
        
        # Optimized greedy selection
        for i, (feat, target_corr) in enumerate(feature_target_pairs):
            if i not in available_indices:
                continue
            if len(final_selected) >= max_total_features:
                break
                
            # Check if uncorrelated with all selected features (vectorized)
            is_uncorrelated = True
            for selected_feat in final_selected:
                if corr_matrix.loc[feat, selected_feat] > corr_threshold:
                    is_uncorrelated = False
                    break
            
            if is_uncorrelated:
                final_selected.append(feat)
                # Remove correlated features from available pool for efficiency
                to_remove = set()
                for j, (other_feat, _) in enumerate(feature_target_pairs):
                    if j in available_indices and corr_matrix.loc[feat, other_feat] > corr_threshold:
                        to_remove.add(j)
                available_indices -= to_remove
    else:
        final_selected = deduplicated  # Use all correlation-filtered features
    
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