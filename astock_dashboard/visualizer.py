"""
A股 Dashboard 可视化组件

提供各类图表和可视化功能
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_pnl_distribution(portfolio_df: pd.DataFrame) -> go.Figure:
    """
    绘制持仓盈亏分布图
    
    Args:
        portfolio_df: 持仓 DataFrame
        
    Returns:
        Plotly Figure
    """
    if portfolio_df.empty:
        return go.Figure()
    
    # 根据盈亏设置颜色
    colors = ['#ef4444' if x < 0 else '#22c55e' for x in portfolio_df['pnl_pct']]
    
    # 显示名称
    if 'name' in portfolio_df.columns:
        labels = portfolio_df['name']
    else:
        labels = portfolio_df['code']
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=portfolio_df['pnl_pct'],
        marker_color=colors,
        text=[f"{x:.2%}" for x in portfolio_df['pnl_pct']],
        textposition='outside'
    )])
    
    fig.update_layout(
        title="持仓盈亏分布",
        yaxis_tickformat='.2%',
        yaxis_title="盈亏比例",
        xaxis_title="股票",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    
    return fig


def plot_market_heatmap(market_df: pd.DataFrame) -> go.Figure:
    """
    绘制市场热力图（涨跌分布）
    
    Args:
        market_df: 市场数据 DataFrame
        
    Returns:
        Plotly Figure
    """
    if market_df.empty:
        return go.Figure()
    
    # 按涨跌幅着色
    colors = market_df['pct_chg'].apply(
        lambda x: '#ef4444' if x < -5 else ('#f97316' if x < -2 else 
                 ('#fbbf24' if x < 0 else ('#84cc16' if x < 2 else 
                 ('#22c55e' if x < 5 else '#16a34a'))))
    )
    
    fig = go.Figure(data=[go.Scatter(
        x=market_df['market_cap'] if 'market_cap' in market_df.columns else range(len(market_df)),
        y=market_df['pct_chg'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=market_df['pct_chg'],
            colorscale='RdYlGn',
            cmin=-10,
            cmax=10,
            showscale=True,
            colorbar=dict(title="涨跌幅%")
        ),
        text=market_df['name'] if 'name' in market_df.columns else market_df['code'],
        textposition='top center',
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "涨跌幅: %{y:.2f}%<br>" +
            "市值: %{x:.2f}亿<br>" +
            "<extra></extra>"
        )
    )])
    
    fig.update_layout(
        title="市场涨跌分布",
        xaxis_title="市值（亿）",
        yaxis_title="涨跌幅 (%)",
        xaxis_type="log",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    
    return fig


def plot_kline_chart(kline_df: pd.DataFrame, name: str = "") -> go.Figure:
    """
    绘制K线图
    
    Args:
        kline_df: K线数据 DataFrame
        name: 股票名称
        
    Returns:
        Plotly Figure
    """
    if kline_df.empty:
        return go.Figure()
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # K线图
    fig.add_trace(go.Candlestick(
        x=kline_df['trade_date'],
        open=kline_df['open'],
        high=kline_df['high'],
        low=kline_df['low'],
        close=kline_df['close'],
        name='K线',
        increasing_line_color='#ef4444',  # 红涨
        decreasing_line_color='#22c55e',  # 绿跌
        increasing_fillcolor='#ef4444',
        decreasing_fillcolor='#22c55e'
    ), row=1, col=1)
    
    # 添加均线
    if len(kline_df) >= 5:
        kline_df['MA5'] = kline_df['close'].rolling(5).mean()
        fig.add_trace(go.Scatter(
            x=kline_df['trade_date'],
            y=kline_df['MA5'],
            name='MA5',
            line=dict(color='#f59e0b', width=1)
        ), row=1, col=1)
    
    if len(kline_df) >= 10:
        kline_df['MA10'] = kline_df['close'].rolling(10).mean()
        fig.add_trace(go.Scatter(
            x=kline_df['trade_date'],
            y=kline_df['MA10'],
            name='MA10',
            line=dict(color='#3b82f6', width=1)
        ), row=1, col=1)
    
    if len(kline_df) >= 20:
        kline_df['MA20'] = kline_df['close'].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=kline_df['trade_date'],
            y=kline_df['MA20'],
            name='MA20',
            line=dict(color='#8b5cf6', width=1)
        ), row=1, col=1)
    
    # 成交量
    colors = ['#ef4444' if kline_df.iloc[i]['close'] >= kline_df.iloc[i]['open'] 
              else '#22c55e' for i in range(len(kline_df))]
    
    fig.add_trace(go.Bar(
        x=kline_df['trade_date'],
        y=kline_df['volume'],
        name='成交量',
        marker_color=colors
    ), row=2, col=1)
    
    fig.update_layout(
        title=f"{name} K线图" if name else "K线图",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    return fig


def plot_training_curve(history_df: pd.DataFrame) -> go.Figure:
    """
    绘制训练曲线
    
    Args:
        history_df: 训练历史 DataFrame
        
    Returns:
        Plotly Figure
    """
    if history_df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("回测得分", "最佳分数")
    )
    
    # 当前得分
    if 'score' in history_df.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(history_df))),
            y=history_df['score'],
            mode='lines',
            name='当前得分',
            line=dict(color='#3b82f6', width=1)
        ), row=1, col=1)
    
    # 最佳得分
    if 'best_score' in history_df.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(history_df))),
            y=history_df['best_score'],
            mode='lines',
            name='最佳得分',
            line=dict(color='#22c55e', width=2)
        ), row=2, col=1)
    
    fig.update_layout(
        title="模型训练曲线",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )
    
    return fig


def plot_sector_performance(sector_df: pd.DataFrame) -> go.Figure:
    """
    绘制板块表现
    
    Args:
        sector_df: 板块统计 DataFrame
        
    Returns:
        Plotly Figure
    """
    if sector_df.empty:
        return go.Figure()
    
    # 取前15个板块
    df = sector_df.head(15)
    
    colors = ['#ef4444' if x < 0 else '#22c55e' for x in df['avg_pct_chg']]
    
    fig = go.Figure(data=[go.Bar(
        x=df['avg_pct_chg'],
        y=df['industry'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.2f}%" for x in df['avg_pct_chg']],
        textposition='outside'
    )])
    
    fig.update_layout(
        title="板块涨跌排行",
        xaxis_title="平均涨跌幅 (%)",
        template="plotly_dark",
        height=450,
        margin=dict(l=100, r=50, t=50, b=20),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_market_breadth(stats: dict) -> go.Figure:
    """
    绘制市场宽度（涨跌家数）
    
    Args:
        stats: 市场统计字典
        
    Returns:
        Plotly Figure
    """
    labels = ['上涨', '下跌', '平盘']
    values = [stats.get('up_count', 0), stats.get('down_count', 0), stats.get('flat_count', 0)]
    colors = ['#22c55e', '#ef4444', '#6b7280']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+value',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="涨跌家数",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    # 中心文字
    total = sum(values)
    fig.add_annotation(
        text=f"共{total}只",
        x=0.5, y=0.5,
        font=dict(size=16, color='white'),
        showarrow=False
    )
    
    return fig


def plot_limit_stats(stats: dict) -> go.Figure:
    """
    绘制涨跌停统计
    
    Args:
        stats: 市场统计字典
        
    Returns:
        Plotly Figure
    """
    labels = ['涨停', '跌停']
    values = [stats.get('limit_up', 0), stats.get('limit_down', 0)]
    colors = ['#ef4444', '#22c55e']
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition='outside'
    )])
    
    fig.update_layout(
        title="涨跌停数量",
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="数量"
    )
    
    return fig


def plot_backtest_equity(backtest_result: dict) -> go.Figure:
    """
    绘制回测收益曲线
    
    Args:
        backtest_result: 回测结果字典
        
    Returns:
        Plotly Figure
    """
    dates = backtest_result.get('dates', [])
    cum_returns = backtest_result.get('cum_returns', [])
    
    if not dates or not cum_returns:
        return go.Figure()
    
    # 转换为百分比
    cum_returns_pct = [r * 100 for r in cum_returns]
    
    fig = go.Figure()
    
    # 收益曲线
    fig.add_trace(go.Scatter(
        x=dates,
        y=cum_returns_pct,
        mode='lines',
        name='累计收益',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    
    # 零线
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    
    fig.update_layout(
        title="策略回测收益曲线",
        xaxis_title="日期",
        yaxis_title="累计收益 (%)",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    
    return fig


def plot_backtest_daily_returns(backtest_result: dict) -> go.Figure:
    """
    绘制每日收益分布
    
    Args:
        backtest_result: 回测结果字典
        
    Returns:
        Plotly Figure
    """
    dates = backtest_result.get('dates', [])
    daily_returns = backtest_result.get('daily_returns', [])
    
    if not dates or not daily_returns:
        return go.Figure()
    
    # 转换为百分比
    daily_returns_pct = [r * 100 for r in daily_returns]
    
    # 根据正负设置颜色
    colors = ['#22c55e' if r >= 0 else '#ef4444' for r in daily_returns_pct]
    
    fig = go.Figure(data=[go.Bar(
        x=dates,
        y=daily_returns_pct,
        marker_color=colors,
        name='每日收益'
    )])
    
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    
    fig.update_layout(
        title="每日收益分布",
        xaxis_title="日期",
        yaxis_title="日收益 (%)",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig
