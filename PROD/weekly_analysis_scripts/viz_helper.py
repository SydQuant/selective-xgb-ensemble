"""
Visualization helpers for portfolio backtest - leveraging xgb_compare visualization patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime


def create_symbol_performance_grid(portfolio_df: pd.DataFrame, daily_metrics: pd.DataFrame,
                                   start_date: datetime, end_date: datetime, output_dir: Path):
    """
    Create individual symbol performance grid with uniform Y-axis scale.
    Shows each symbol's cumulative PnL in a grid layout.
    """
    symbols = sorted(portfolio_df['symbol'].unique())
    n_symbols = len(symbols)

    # Calculate grid size
    if n_symbols <= 6:
        rows, cols = 2, 3
    elif n_symbols <= 12:
        rows, cols = 3, 4
    elif n_symbols <= 20:
        rows, cols = 4, 5
    elif n_symbols <= 30:
        rows, cols = 5, 6
    else:
        rows, cols = 6, 6

    # Larger subplots
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows))
    fig.suptitle(f'Individual Symbol Performance: {start_date.date()} to {end_date.date()}',
                fontsize=18, fontweight='bold')

    if n_symbols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Calculate global Y-axis limits using raw returns
    all_pnls = []
    for symbol in symbols:
        symbol_df = portfolio_df[portfolio_df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date')
        cumulative_pnl = symbol_df['return_pct'].cumsum()
        all_pnls.extend(cumulative_pnl.values)

    y_min = min(all_pnls) * 1.1 if min(all_pnls) < 0 else min(all_pnls) * 0.9
    y_max = max(all_pnls) * 1.1 if max(all_pnls) > 0 else max(all_pnls) * 0.9

    # Calculate max absolute PnL for line thickness scaling
    max_abs_pnl = max(abs(y_min), abs(y_max))

    for i, symbol in enumerate(symbols):
        ax = axes[i]
        symbol_df = portfolio_df[portfolio_df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date')

        # Calculate cumulative raw return (not weighted) for symbol
        cumulative_pnl = symbol_df['return_pct'].cumsum()
        dates = pd.to_datetime(symbol_df['date'])

        # Stats
        total_pnl = cumulative_pnl.iloc[-1]
        avg_weight = symbol_df['weight_pct'].mean()

        # Scale line thickness based on magnitude (larger movers = thicker lines)
        linewidth = 2 + 3 * (abs(total_pnl) / max_abs_pnl) if max_abs_pnl > 0 else 2

        # Plot with scaled line thickness
        ax.plot(dates, cumulative_pnl, linewidth=linewidth, color='darkblue', alpha=0.8)
        ax.fill_between(dates, 0, cumulative_pnl, alpha=0.3,
                       color='green' if total_pnl > 0 else 'red')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Add end value label on the plot
        if len(dates) > 0:
            ax.text(dates.iloc[-1], cumulative_pnl.iloc[-1], f'  {total_pnl:.1f}%',
                   fontsize=10, fontweight='bold', va='center',
                   color='darkgreen' if total_pnl > 0 else 'darkred')

        ax.set_title(f'{symbol}\nPnL: {total_pnl:.2f}% | Wt: {avg_weight:.1f}%',
                    fontsize=11, fontweight='bold')
        ax.set_ylim(y_min, y_max)  # Uniform scale
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=9)

    # Hide unused subplots
    for j in range(len(symbols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    save_path = output_dir / f"symbol_grid_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def create_performance_and_metrics(portfolio_df: pd.DataFrame, daily_metrics: pd.DataFrame,
                                   output_dir: Path, start_date: datetime, end_date: datetime):
    """
    Combined performance and metrics: portfolio/baskets, drawdown, rolling Sharpe, risk-return.
    """
    baskets = sorted(portfolio_df['basket'].unique())

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    dates = pd.to_datetime(daily_metrics['date'])
    cumulative_pnl = daily_metrics['cumulative_return_pct'].values
    returns = daily_metrics['portfolio_return_pct'].values

    # Row 1: Portfolio & Basket Performance (top, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, cumulative_pnl, linewidth=3, color='black', label='Portfolio Total', alpha=0.8)
    for basket in baskets:
        basket_df = portfolio_df[portfolio_df['basket'] == basket].copy()
        daily_basket = basket_df.groupby('date')['weighted_return_pct'].sum().reset_index()
        daily_basket = daily_basket.sort_values('date')
        cumulative = daily_basket['weighted_return_pct'].cumsum()
        ax1.plot(pd.to_datetime(daily_basket['date']), cumulative, linewidth=2, label=basket, marker='o', markersize=3, alpha=0.7)
    ax1.set_title('Portfolio & Basket Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10, ncol=3)

    # Row 2, Col 1: Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    running_max = pd.Series(cumulative_pnl).expanding().max()
    drawdown = cumulative_pnl - running_max
    ax2.fill_between(dates, drawdown, 0, alpha=0.4, color='red', label='Drawdown')
    ax2.plot(dates, drawdown, color='darkred', linewidth=1.5)
    ax2.set_title('Drawdown', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Row 2, Col 2: 30-day Rolling Sharpe (skip first 30 days)
    ax3 = fig.add_subplot(gs[1, 1])
    window = 30
    rolling_sharpe = pd.Series(returns).rolling(window).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    valid_dates = dates[window:]
    valid_sharpe = rolling_sharpe.values[window:]
    ax3.plot(valid_dates, valid_sharpe, linewidth=2, color='green')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=1, color='blue', linestyle=':', alpha=0.5, label='Sharpe=1')
    ax3.set_title('30-Day Rolling Sharpe', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Row 3: Risk-Return Profile (bottom, spans both columns)
    ax4 = fig.add_subplot(gs[2, :])
    symbol_stats = []
    for symbol in portfolio_df['symbol'].unique():
        sym_df = portfolio_df[portfolio_df['symbol'] == symbol]
        daily_returns = sym_df.groupby('date')['weighted_return_pct'].sum()
        total_return = daily_returns.sum()
        vol = daily_returns.std() * np.sqrt(252) * 100  # Annualized
        symbol_stats.append({'symbol': symbol, 'return': total_return, 'vol': vol})

    stats_df = pd.DataFrame(symbol_stats)
    ax4.scatter(stats_df['vol'], stats_df['return'], s=200, alpha=0.6, edgecolors='black', linewidths=2)

    # Add labels with offset to reduce overlap
    for _, row in stats_df.iterrows():
        ax4.annotate(row['symbol'], (row['vol'], row['return']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=11, fontweight='bold', alpha=0.9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Risk-Return Profile by Symbol', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Volatility (% annualized)', fontsize=12)
    ax4.set_ylabel('Total Return (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=11)

    plt.suptitle(f'Portfolio Analysis: {start_date.date()} to {end_date.date()}',
                fontsize=16, fontweight='bold', y=0.995)

    save_path = output_dir / f"performance_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def create_metrics_panel(daily_metrics: pd.DataFrame, portfolio_df: pd.DataFrame,
                        output_dir: Path, start_date: datetime, end_date: datetime):
    """
    Drawdown, rolling Sharpe, and risk-return profile.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    dates = pd.to_datetime(daily_metrics['date'])
    returns = daily_metrics['portfolio_return_pct'].values
    cumulative = daily_metrics['cumulative_return_pct'].values

    # Plot 1: Drawdown (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    running_max = pd.Series(cumulative).expanding().max()
    drawdown = cumulative - running_max
    ax1.fill_between(dates, drawdown, 0, alpha=0.4, color='red', label='Drawdown')
    ax1.plot(dates, drawdown, color='darkred', linewidth=1.5)
    ax1.set_title('Drawdown', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Drawdown (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: 30-day Rolling Sharpe (top right, skip first 30 days)
    ax2 = fig.add_subplot(gs[0, 1])
    window = 30
    rolling_sharpe = pd.Series(returns).rolling(window).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    # Skip first 30 days
    valid_dates = dates[window:]
    valid_sharpe = rolling_sharpe.values[window:]
    ax2.plot(valid_dates, valid_sharpe, linewidth=2, color='green')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='blue', linestyle=':', alpha=0.5, label='Sharpe=1')
    ax2.set_title('30-Day Rolling Sharpe', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Plot 3: Risk-Return Profile (bottom, spans both columns)
    ax3 = fig.add_subplot(gs[1, :])
    symbol_stats = []
    for symbol in portfolio_df['symbol'].unique():
        sym_df = portfolio_df[portfolio_df['symbol'] == symbol]
        daily_returns = sym_df.groupby('date')['weighted_return_pct'].sum()
        total_return = daily_returns.sum()
        vol = daily_returns.std() * np.sqrt(252) * 100  # Annualized
        symbol_stats.append({'symbol': symbol, 'return': total_return, 'vol': vol})

    stats_df = pd.DataFrame(symbol_stats)
    ax3.scatter(stats_df['vol'], stats_df['return'], s=150, alpha=0.6, edgecolors='black', linewidths=2)
    for _, row in stats_df.iterrows():
        ax3.annotate(row['symbol'], (row['vol'], row['return']), fontsize=12, fontweight='bold', alpha=0.8)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('Risk-Return Profile by Symbol', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Volatility (% annualized)', fontsize=12)
    ax3.set_ylabel('Total Return (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)

    plt.suptitle(f'Portfolio Metrics: {start_date.date()} to {end_date.date()}',
                fontsize=16, fontweight='bold', y=0.995)

    save_path = output_dir / f"metrics_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path
