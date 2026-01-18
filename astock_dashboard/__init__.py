"""
A股 Dashboard 模块

提供 Streamlit 可视化监控界面
"""
from .data_service import AStockDashboardService
from .visualizer import (
    plot_pnl_distribution,
    plot_market_heatmap,
    plot_kline_chart,
    plot_training_curve,
    plot_sector_performance,
    plot_market_breadth,
    plot_limit_stats
)

__all__ = [
    'AStockDashboardService',
    'plot_pnl_distribution',
    'plot_market_heatmap', 
    'plot_kline_chart',
    'plot_training_curve',
    'plot_sector_performance',
    'plot_market_breadth',
    'plot_limit_stats'
]

