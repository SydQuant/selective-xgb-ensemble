"""
Comprehensive performance reporting for trading strategies.
Calculates key metrics like Sharpe, drawdown, win rate, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_returns_metrics(signal: pd.Series, target_returns: pd.Series, 
                            freq: int = 252) -> Dict[str, float]:
    """
    Calculate core performance metrics - simplified version.
    
    Args:
        signal: Trading signal (positions)
        target_returns: Forward returns to trade on
        freq: Annualization factor (252 for daily, ~2000 for hourly)
    """
    # Align signal and returns
    aligned_signal = signal.shift(1).fillna(0.0)  # Lag signal by 1 period
    aligned_returns = target_returns.reindex_like(aligned_signal).fillna(0.0)
    
    # Calculate strategy PnL
    strategy_returns = (aligned_signal * aligned_returns).dropna()
    
    if len(strategy_returns) == 0:
        return _empty_metrics()
    
    # Cumulative metrics
    cumulative_returns = strategy_returns.cumsum()
    equity_curve = (1 + strategy_returns).cumprod()
    
    # Basic statistics
    total_return = float(cumulative_returns.iloc[-1])
    annualized_return = float(strategy_returns.mean() * freq)
    volatility = float(strategy_returns.std() * np.sqrt(freq))
    
    # Risk-adjusted metrics
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
    
    # Drawdown calculation
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(drawdown.min())
    
    # Win metrics
    positive_returns = strategy_returns[strategy_returns > 0]
    negative_returns = strategy_returns[strategy_returns < 0]
    
    win_rate = len(positive_returns) / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
    avg_win = float(positive_returns.mean()) if len(positive_returns) > 0 else 0.0
    avg_loss = float(negative_returns.mean()) if len(negative_returns) > 0 else 0.0
    
    # Hit rate (signal accuracy)
    correct_signals = ((aligned_signal > 0) & (aligned_returns > 0)) | \
                     ((aligned_signal < 0) & (aligned_returns < 0))
    hit_rate = float(correct_signals.mean()) if len(correct_signals) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'hit_rate': hit_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'num_trades': len(strategy_returns),
        'num_wins': len(positive_returns),
        'num_losses': len(negative_returns)
    }

def _empty_metrics() -> Dict[str, float]:
    """Return empty metrics dict when no data available."""
    return {k: 0.0 for k in [
        'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
        'max_drawdown', 'win_rate', 'hit_rate', 'avg_win', 'avg_loss',
        'num_trades', 'num_wins', 'num_losses'
    ]}

def analyze_symbol_contributions(signal: pd.Series, target_returns: pd.Series,
                               symbol_signals: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    """
    Analyze individual symbol contributions to overall performance.
    
    Args:
        signal: Combined ensemble signal
        target_returns: Target returns
        symbol_signals: Dict of individual symbol signals {symbol: signal}
    """
    results = {}
    
    # Overall strategy performance
    overall_metrics = calculate_returns_metrics(signal, target_returns)
    results['ENSEMBLE'] = overall_metrics
    
    # Individual symbol analysis
    for symbol, sym_signal in symbol_signals.items():
        if sym_signal is not None and len(sym_signal) > 0:
            # Calculate individual symbol performance
            sym_metrics = calculate_returns_metrics(sym_signal, target_returns)
            results[symbol] = sym_metrics
        else:
            results[symbol] = _empty_metrics()
    
    return results

def format_performance_report(metrics: Dict[str, float], title: str = "Performance Report") -> str:
    """Format core performance metrics into a readable report."""
    
    report = [
        f"\n{'='*50}",
        f"{title:^50}",
        f"{'='*50}",
        "",
        f"📈 RETURNS:",
        f"  Total Return:        {metrics['total_return']:>8.2%}",
        f"  Annualized Return:   {metrics['annualized_return']:>8.2%}",
        f"  Volatility:          {metrics['volatility']:>8.2%}",
        f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}",
        "",
        f"📉 RISK:",
        f"  Max Drawdown:        {metrics['max_drawdown']:>8.2%}",
        "",
        f"🎯 ACCURACY:",
        f"  Win Rate:            {metrics['win_rate']:>8.2%}",
        f"  Hit Rate:            {metrics['hit_rate']:>8.2%}",
        f"  Avg Win:             {metrics['avg_win']:>8.4f}",
        f"  Avg Loss:            {metrics['avg_loss']:>8.4f}",
        "",
        f"📊 TRADES:",
        f"  Total Trades:        {metrics['num_trades']:>8.0f}",
        f"  Winning Trades:      {metrics['num_wins']:>8.0f}",
        f"  Losing Trades:       {metrics['num_losses']:>8.0f}",
        f"{'='*50}"
    ]
    
    return '\n'.join(report)

def format_symbol_breakdown(symbol_results: Dict[str, Dict[str, float]]) -> str:
    """Format per-symbol performance breakdown."""
    
    report = [
        f"\n{'='*75}",
        f"{'SYMBOL PERFORMANCE BREAKDOWN':^75}",
        f"{'='*75}",
        "",
        f"{'Symbol':<10} {'Return':<8} {'Sharpe':<7} {'MaxDD':<7} {'WinRate':<8} {'HitRate':<8} {'Trades':<6}"
    ]
    
    # Sort symbols by Sharpe ratio (descending)
    sorted_symbols = sorted(symbol_results.items(), 
                          key=lambda x: x[1]['sharpe_ratio'], 
                          reverse=True)
    
    for symbol, metrics in sorted_symbols:
        symbol_display = symbol if len(symbol) <= 9 else symbol[:9]
        report.append(
            f"{symbol_display:<10} "
            f"{metrics['total_return']:<7.1%} "
            f"{metrics['sharpe_ratio']:<6.2f} "
            f"{metrics['max_drawdown']:<6.1%} "
            f"{metrics['win_rate']:<7.1%} "
            f"{metrics['hit_rate']:<7.1%} "
            f"{metrics['num_trades']:<6.0f}"
        )
    
    report.extend([
        "",
        f"{'='*75}"
    ])
    
    return '\n'.join(report)

def save_performance_csv(metrics: Dict[str, float], symbol_results: Dict[str, Dict[str, float]], 
                        filepath: str = "artifacts/performance_summary.csv"):
    """Save performance metrics to CSV for further analysis."""
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create comprehensive DataFrame
    all_results = {'ENSEMBLE': metrics}
    all_results.update(symbol_results)
    
    df = pd.DataFrame(all_results).T
    df.index.name = 'Symbol'
    df.to_csv(filepath)
    logger.info(f"Performance summary saved to {filepath}")
    
    return df