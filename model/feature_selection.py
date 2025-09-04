"""
Block-wise feature selection for large feature sets.
This preprocessing step selects the best features across blocks, 
then all XGB models use the same selected feature set.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import logging
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)

def block_wise_feature_selection(X: pd.DataFrame, y: pd.Series, 
                                block_size: int = 100, 
                                features_per_block: int = 20,
                                max_total_features: int = 50,  # Default to 50, can override for 70/100
                                corr_threshold: float = 0.7,
                                diversity_weight: float = 0.3) -> List[str]:
    """
    Cleverer block-wise feature selection that approximates clustering results:
    1. Split features into blocks
    2. For each block, select best features AND remove highly correlated ones (local clustering)
    3. Combine and do final correlation-based deduplication (global clustering)
    4. Final ranking by target correlation
    
    This should approximate full clustering but be much faster on large feature sets.
    """
    feature_names = X.columns.tolist()
    
    if len(feature_names) <= max_total_features:
        logger.info(f"Feature count ({len(feature_names)}) <= max ({max_total_features}), using all features")
        return feature_names
    
    logger.info(f"Smart block-wise selection: {len(feature_names)} -> blocks of {block_size} -> local clustering -> global deduplication")
    
    # Step 1: Create blocks
    blocks = []
    for i in range(0, len(feature_names), block_size):
        block = feature_names[i:i + block_size]
        blocks.append(block)
    
    # Step 2: For each block, select best features + local clustering
    all_candidates = []
    
    for i, block in enumerate(blocks):
        logger.info(f"Processing block {i+1}/{len(blocks)}: {len(block)} features")
        
        # Get block data
        X_block = X[block]
        
        # Remove features with zero variance
        non_zero_var = X_block.std() > 1e-10
        valid_features = X_block.columns[non_zero_var].tolist()
        
        if len(valid_features) == 0:
            logger.warning(f"Block {i+1}: No valid features (zero variance)")
            continue
        
        X_block = X_block[valid_features]
        
        # Calculate both target correlation and mutual information for diversity
        target_correlations = {}
        mutual_info_scores = {}
        
        for feat in valid_features:
            try:
                corr = X_block[feat].corr(y)
                target_correlations[feat] = abs(corr) if pd.notna(corr) else 0.0
            except:
                target_correlations[feat] = 0.0
        
        # Calculate mutual information for non-linear relationships
        try:
            mi_scores = mutual_info_regression(X_block[valid_features].values, y.values, random_state=42)
            for i, feat in enumerate(valid_features):
                mutual_info_scores[feat] = mi_scores[i] if not np.isnan(mi_scores[i]) else 0.0
        except:
            # Fallback to correlation if MI fails
            mutual_info_scores = target_correlations.copy()
        
        # Combine correlation and mutual information for better representativeness
        combined_scores = {}
        for feat in valid_features:
            # Normalize scores to [0,1] within block
            max_corr = max(target_correlations.values()) if target_correlations.values() else 1.0
            max_mi = max(mutual_info_scores.values()) if mutual_info_scores.values() else 1.0
            
            norm_corr = target_correlations[feat] / max_corr if max_corr > 0 else 0
            norm_mi = mutual_info_scores[feat] / max_mi if max_mi > 0 else 0
            
            # Weighted combination: correlation + mutual information for diversity
            combined_scores[feat] = (1 - diversity_weight) * norm_corr + diversity_weight * norm_mi
        
        # Sort by combined score for better representativeness
        ranked_features = sorted(valid_features, key=lambda x: combined_scores[x], reverse=True)
        
        # Local clustering: remove highly correlated features within block
        block_selected = []
        for feat in ranked_features:
            # Check correlation with already selected features in this block
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
                if len(block_selected) >= features_per_block:
                    break
        
        all_candidates.extend(block_selected)
        logger.info(f"Block {i+1}: Selected {len(block_selected)} features (best: {target_correlations[ranked_features[0]]:.4f})")
    
    # Step 3: Global deduplication - remove correlated features across blocks
    logger.info(f"Global deduplication: {len(all_candidates)} candidates -> removing cross-block correlations")
    
    # Recalculate target correlations for all candidates
    final_correlations = {}
    for feat in all_candidates:
        try:
            corr = X[feat].corr(y)
            final_correlations[feat] = abs(corr) if pd.notna(corr) else 0.0
        except:
            final_correlations[feat] = 0.0
    
    # Sort all candidates by target correlation
    ranked_candidates = sorted(all_candidates, key=lambda x: final_correlations[x], reverse=True)
    
    # Global clustering: keep best features, remove highly correlated ones
    # Use adaptive correlation threshold based on target feature count
    if max_total_features >= 100:
        global_corr_threshold = 0.85  # More permissive for high feature counts
    elif max_total_features >= 70:
        global_corr_threshold = 0.8   # Moderately permissive
    else:
        global_corr_threshold = corr_threshold  # Original threshold for lower counts
    
    final_selected = []
    for feat in ranked_candidates:
        if len(final_selected) >= max_total_features:
            break
            
        # Check correlation with already selected features globally  
        keep_feature = True
        for selected_feat in final_selected:
            try:
                feat_corr = X[feat].corr(X[selected_feat])
                if pd.notna(feat_corr) and abs(feat_corr) > global_corr_threshold:
                    keep_feature = False
                    break
            except:
                pass
        
        if keep_feature:
            final_selected.append(feat)
    
    logger.info(f"✅ Smart block-wise selection complete: {len(final_selected)} features selected")
    logger.info(f"   Local clustering: {corr_threshold}, Global clustering: {global_corr_threshold}")
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