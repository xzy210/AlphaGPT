"""
A股 AlphaGPT 监控面板

基于 Streamlit 的可视化监控界面
"""
import streamlit as st
import pandas as pd
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astock_dashboard.data_service import AStockDashboardService
from astock_dashboard.visualizer import (
    plot_pnl_distribution, plot_market_heatmap, plot_kline_chart,
    plot_training_curve, plot_sector_performance, plot_market_breadth,
    plot_limit_stats, plot_backtest_equity, plot_backtest_daily_returns
)

# 页面配置
st.set_page_config(
    page_title="A股 AlphaGPT 监控",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    /* 主题色 */
    :root {
        --primary-color: #3b82f6;
        --success-color: #22c55e;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
    }
    
    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* 涨跌颜色 */
    .up { color: #ef4444 !important; }
    .down { color: #22c55e !important; }
    
    /* 数据表格 */
    .stDataFrame { border: none !important; }
    
    /* 侧边栏 */
    .css-1d391kg { background-color: #0f172a; }
    
    /* 标题 */
    h1, h2, h3 { color: #e2e8f0 !important; }
    
    /* 按钮 */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* 紧急停止按钮 */
    .emergency-btn>button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    """获取数据服务（缓存）"""
    return AStockDashboardService()


def format_number(num, suffix=''):
    """格式化数字"""
    if num >= 100000000:
        return f"{num/100000000:.2f}亿{suffix}"
    elif num >= 10000:
        return f"{num/10000:.2f}万{suffix}"
    else:
        return f"{num:.2f}{suffix}"


def main():
    svc = get_service()
    
    # ============== 侧边栏 ==============
    st.sidebar.title("📈 A股 AlphaGPT")
    st.sidebar.markdown("---")
    
    # 账户信息
    with st.sidebar:
        st.subheader("💰 账户状态")
        account = svc.get_account_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总资产", format_number(account['total_asset'], '元'))
        with col2:
            st.metric("可用资金", format_number(account['cash'], '元'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("持仓市值", format_number(account['market_value'], '元'))
        with col2:
            profit_pct = account.get('profit_pct', 0)
            st.metric("收益率", f"{profit_pct:.2%}", 
                     delta=f"{profit_pct:.2%}" if profit_pct != 0 else None)
        
        st.markdown("---")
        
        # 数据库状态
        st.subheader("🗄️ 数据库状态")
        db_stats = svc.get_db_stats()
        st.caption(f"股票数量: {db_stats['stock_count']}")
        st.caption(f"K线记录: {db_stats['kline_records']:,}")
        st.caption(f"数据范围: {db_stats['min_date']} ~ {db_stats['max_date']}")
        
        st.markdown("---")
        
        # 控制面板
        st.subheader("🎮 控制面板")
        if st.button("🔄 刷新数据", width="stretch"):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("")
        
        if st.button("🛑 紧急停止", width="stretch", type="primary"):
            with open("ASTOCK_STOP_SIGNAL", "w") as f:
                f.write("STOP")
            st.error("⚠️ 停止信号已发送！策略将在下个周期终止。")
    
    # ============== 主内容区 ==============
    
    # 顶部指标卡片
    market_stats = svc.get_market_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📅 数据日期",
            market_stats['latest_date'] or "无数据"
        )
    
    with col2:
        up_pct = market_stats['up_count'] / max(market_stats['total_stocks'], 1) * 100
        st.metric(
            "📈 上涨家数",
            f"{market_stats['up_count']}",
            delta=f"{up_pct:.1f}%"
        )
    
    with col3:
        down_pct = market_stats['down_count'] / max(market_stats['total_stocks'], 1) * 100
        st.metric(
            "📉 下跌家数",
            f"{market_stats['down_count']}",
            delta=f"-{down_pct:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "🔴 涨停",
            f"{market_stats['limit_up']}",
            delta="涨停板"
        )
    
    with col5:
        st.metric(
            "🟢 跌停",
            f"{market_stats['limit_down']}",
            delta="跌停板",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # 标签页
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 持仓监控", "🌍 市场概览", "📈 个股K线", "🧠 策略训练", "📉 策略回测", "📝 系统日志"
    ])
    
    # ============== Tab 1: 持仓监控 ==============
    with tab1:
        st.subheader("📊 当前持仓")
        
        portfolio_df = svc.load_portfolio()
        
        if not portfolio_df.empty:
            # 持仓汇总
            total_value = portfolio_df.get('market_value', portfolio_df.get('amount', 0) * portfolio_df.get('current_price', 0)).sum()
            total_pnl = portfolio_df.get('pnl_amount', 0).sum() if 'pnl_amount' in portfolio_df.columns else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("持仓数量", f"{len(portfolio_df)} 只")
            with col2:
                st.metric("持仓市值", format_number(total_value, '元'))
            with col3:
                st.metric("持仓盈亏", format_number(total_pnl, '元'),
                         delta=f"{total_pnl/max(total_value-total_pnl, 1):.2%}" if total_value > 0 else None)
            
            st.markdown("")
            
            # 持仓表格
            display_cols = [c for c in ['code', 'name', 'entry_price', 'current_price', 
                                        'amount', 'pnl_pct', 'pnl_amount'] 
                          if c in portfolio_df.columns]
            
            if display_cols:
                show_df = portfolio_df[display_cols].copy()
                
                # 格式化
                if 'pnl_pct' in show_df.columns:
                    show_df['pnl_pct'] = show_df['pnl_pct'].apply(lambda x: f"{x:.2%}")
                if 'entry_price' in show_df.columns:
                    show_df['entry_price'] = show_df['entry_price'].apply(lambda x: f"{x:.3f}")
                if 'current_price' in show_df.columns:
                    show_df['current_price'] = show_df['current_price'].apply(lambda x: f"{x:.3f}")
                
                # 重命名列
                col_names = {
                    'code': '股票代码', 'name': '股票名称',
                    'entry_price': '买入价', 'current_price': '现价',
                    'amount': '持仓数量', 'pnl_pct': '盈亏比例', 'pnl_amount': '盈亏金额'
                }
                show_df = show_df.rename(columns=col_names)
                
                st.dataframe(show_df, width="stretch", hide_index=True)
            
            st.markdown("")
            
            # 盈亏分布图
            if 'pnl_pct' in portfolio_df.columns:
                st.plotly_chart(plot_pnl_distribution(portfolio_df), key="pnl_dist")
        else:
            st.info("📭 暂无持仓，策略正在扫描买入机会...")
    
    # ============== Tab 2: 市场概览 ==============
    with tab2:
        st.subheader("🌍 市场概览")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 涨跌家数饼图
            st.plotly_chart(plot_market_breadth(market_stats), key="breadth")
        
        with col2:
            # 涨跌停统计
            st.plotly_chart(plot_limit_stats(market_stats), key="limit")
        
        st.markdown("---")
        
        # 板块表现
        st.subheader("🏭 板块表现")
        sector_df = svc.get_sector_stats()
        if not sector_df.empty:
            st.plotly_chart(plot_sector_performance(sector_df), key="sector")
        else:
            st.warning("暂无板块数据")
        
        st.markdown("---")
        
        # 涨幅榜
        st.subheader("🔥 涨幅榜 TOP 20")
        market_df = svc.get_market_overview(limit=20)
        
        if not market_df.empty:
            # 格式化表格
            show_df = market_df.copy()
            show_df['pct_chg'] = show_df['pct_chg'].apply(lambda x: f"{x:+.2f}%")
            show_df['close'] = show_df['close'].apply(lambda x: f"{x:.2f}")
            show_df['amount'] = show_df['amount'].apply(lambda x: format_number(x, ''))
            show_df['market_cap'] = show_df['market_cap'].apply(
                lambda x: format_number(x, '') if pd.notna(x) else '-'
            )
            
            display_cols = ['code', 'name', 'close', 'pct_chg', 'amount', 'turnover', 'market_cap']
            display_cols = [c for c in display_cols if c in show_df.columns]
            
            col_names = {
                'code': '代码', 'name': '名称', 'close': '收盘价',
                'pct_chg': '涨跌幅', 'amount': '成交额', 'turnover': '换手率',
                'market_cap': '市值'
            }
            
            st.dataframe(
                show_df[display_cols].rename(columns=col_names),
                width="stretch",
                hide_index=True
            )
        else:
            st.warning("暂无市场数据，请先运行数据管线")
    
    # ============== Tab 3: 个股K线 ==============
    with tab3:
        st.subheader("📈 个股K线")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 股票搜索
            search_keyword = st.text_input(
                "搜索股票",
                placeholder="输入股票代码或名称，如: 000001 或 平安银行"
            )
        
        with col2:
            days = st.selectbox("K线周期", [30, 60, 120, 250], index=1)
        
        selected_code = None
        selected_name = ""
        
        if search_keyword:
            search_result = svc.search_stock(search_keyword)
            if not search_result.empty:
                # 显示搜索结果
                options = [f"{row['code']} - {row['name']}" for _, row in search_result.iterrows()]
                selected = st.selectbox("选择股票", options)
                if selected:
                    selected_code = selected.split(' - ')[0]
                    selected_name = selected.split(' - ')[1]
            else:
                st.warning("未找到匹配的股票")
        
        if selected_code:
            kline_df = svc.get_kline_data(selected_code, days)
            if not kline_df.empty:
                st.plotly_chart(
                    plot_kline_chart(kline_df, f"{selected_code} {selected_name}"),
                    key="kline"
                )
                
                # 显示最新数据
                latest = kline_df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最新收盘", f"{latest['close']:.2f}")
                with col2:
                    st.metric("最高价", f"{latest['high']:.2f}")
                with col3:
                    st.metric("最低价", f"{latest['low']:.2f}")
                with col4:
                    st.metric("成交量", format_number(latest['volume'], '手'))
            else:
                st.warning("暂无该股票K线数据")
    
    # ============== Tab 4: 策略训练 ==============
    with tab4:
        st.subheader("🧠 策略训练")
        
        # 当前策略
        strategy = svc.load_strategy_info()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📌 当前最优策略")
            st.code(strategy.get('formula_str', '未训练'), language='text')
        
        with col2:
            st.metric("策略得分", f"{strategy.get('score', 0):.4f}")
        
        st.markdown("---")
        
        # 训练历史
        st.markdown("#### 📊 训练曲线")
        history_df = svc.load_training_history()
        
        if not history_df.empty:
            st.plotly_chart(plot_training_curve(history_df), key="training")
            
            # 训练统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("训练轮次", len(history_df))
            with col2:
                if 'best_score' in history_df.columns:
                    st.metric("最佳得分", f"{history_df['best_score'].max():.4f}")
            with col3:
                if 'score' in history_df.columns:
                    st.metric("最近得分", f"{history_df['score'].iloc[-1]:.4f}")
        else:
            st.info("📭 暂无训练历史，请先运行训练脚本: `python train_astock.py`")
    
    # ============== Tab 5: 策略回测 ==============
    with tab5:
        st.subheader("📉 策略回测")
        
        # 当前策略
        strategy = svc.load_strategy_info()
        
        st.markdown("#### 📌 回测策略")
        st.code(strategy.get('formula_str', '未训练'), language='text')
        
        st.markdown("---")
        
        # 回测参数
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bt_lookback = st.selectbox("回测天数", [30, 60, 120, 180, 250], index=1)
        
        with col2:
            bt_stocks = st.selectbox("股票数量", [100, 200, 300, 500], index=1)
        
        with col3:
            st.markdown("")
            st.markdown("")
            run_backtest = st.button("🚀 运行回测", type="primary")
        
        # 运行回测
        if run_backtest:
            with st.spinner("正在运行回测，请稍候..."):
                bt_result = svc.run_backtest(lookback_days=bt_lookback, limit_stocks=bt_stocks)
            
            if bt_result.get('error'):
                st.error(f"❌ 回测失败: {bt_result['error']}")
            else:
                st.success("✅ 回测完成!")
                
                # 保存结果到 session state
                st.session_state['backtest_result'] = bt_result
        
        # 显示回测结果
        if 'backtest_result' in st.session_state:
            bt_result = st.session_state['backtest_result']
            
            st.markdown("---")
            st.markdown("#### 📊 回测结果")
            
            # 关键指标
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_ret = bt_result.get('total_return', 0)
                st.metric(
                    "总收益率", 
                    f"{total_ret:.2%}",
                    delta=f"{'盈利' if total_ret > 0 else '亏损'}"
                )
            
            with col2:
                st.metric("回测得分", f"{bt_result.get('score', 0):.4f}")
            
            with col3:
                st.metric("胜率", f"{bt_result.get('win_rate', 0):.2%}")
            
            with col4:
                st.metric("盈亏比", f"{bt_result.get('profit_loss_ratio', 0):.2f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("总交易次数", f"{bt_result.get('total_trades', 0):.0f}")
            
            with col2:
                st.metric("平均仓位", f"{bt_result.get('avg_position', 0):.2%}")
            
            st.markdown("---")
            
            # 收益曲线
            st.plotly_chart(plot_backtest_equity(bt_result), use_container_width=True, key="bt_equity")
            
            # 每日收益
            st.plotly_chart(plot_backtest_daily_returns(bt_result), use_container_width=True, key="bt_daily")
            
            # 回测参数
            st.markdown("---")
            st.caption(f"回测参数: {bt_result.get('stock_count', 0)} 只股票, {bt_result.get('lookback_days', 0)} 天数据")
    
    # ============== Tab 6: 系统日志 ==============
    with tab6:
        st.subheader("📝 系统日志")
        
        log_lines = st.slider("显示行数", 10, 100, 30)
        logs = svc.get_recent_logs(log_lines)
        
        if logs:
            st.code("".join(logs), language="text")
        else:
            st.info("📭 暂无日志文件")
            st.caption("日志文件路径: `astock_strategy.log`")
    
    # ============== 自动刷新 ==============
    st.markdown("---")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("⏱️ 自动刷新", value=False)
    with col2:
        refresh_interval = st.slider("刷新间隔(秒)", 10, 120, 30, disabled=not auto_refresh)
    
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()

