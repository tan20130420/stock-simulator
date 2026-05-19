import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import akshare as ak
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="智能波段交易模拟器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 数据获取模块 ====================

def get_stock_data(symbol, start_date, end_date, market_type="A股"):
    """获取股票/转债/基金历史数据"""
    try:
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        if market_type == "A股":
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                   start_date=start_str, end_date=end_str, adjust="qfq")
        elif market_type == "可转债":
            df = ak.bond_zh_hs_cov_daily(symbol=symbol)
            df = df[(df.index >= start_date.strftime("%Y-%m-%d")) & 
                    (df.index <= end_date.strftime("%Y-%m-%d"))]
            df = df.reset_index()
            df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
        elif market_type == "ETF基金":
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                    start_date=start_str, end_date=end_str, adjust="qfq")
        elif market_type == "场内基金":
            df = ak.fund_lof_hist_em(symbol=symbol, period="daily",
                                    start_date=start_str, end_date=end_str, adjust="qfq")
        else:
            return None

        if df is None or df.empty:
            return None

        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume"
        })

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        return df[['date', 'open', 'close', 'high', 'low', 'volume']]

    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None

def get_stock_name(symbol, market_type="A股"):
    """获取股票名称"""
    try:
        if market_type == "A股":
            df = ak.stock_zh_a_spot_em()
            name = df[df['代码'] == symbol]['名称'].values
            return name[0] if len(name) > 0 else symbol
        elif market_type == "可转债":
            df = ak.bond_zh_hs_cov_spot()
            name = df[df['代码'] == symbol]['名称'].values
            return name[0] if len(name) > 0 else symbol
        elif market_type in ["ETF基金", "场内基金"]:
            df = ak.fund_etf_spot_em()
            name = df[df['代码'] == symbol]['名称'].values
            return name[0] if len(name) > 0 else symbol
        return symbol
    except:
        return symbol

# ==================== 策略引擎 ====================

class TradingStrategy:
    def __init__(self, initial_capital=100000, num_parts=3, buy_fee_rate=0.0003, 
                 sell_fee_rate=0.0013, min_fee=5):
        self.initial_capital = initial_capital
        self.num_parts = num_parts
        self.part_capital = initial_capital / num_parts
        self.buy_fee_rate = buy_fee_rate
        self.sell_fee_rate = sell_fee_rate
        self.min_fee = min_fee

    def calculate_buy_shares(self, price, capital):
        """计算可买入股数（100股整数倍）"""
        max_shares = int(capital / price)
        shares = (max_shares // 100) * 100
        if shares < 100:
            return 0
        return shares

    def calculate_buy_fee(self, amount):
        """计算买入手续费"""
        fee = max(amount * self.buy_fee_rate, self.min_fee)
        return fee

    def calculate_sell_fee(self, amount):
        """计算卖出手续费"""
        fee = max(amount * self.sell_fee_rate, self.min_fee)
        return fee

    def backtest(self, df, x_percent, y_percent):
        """回测策略"""
        if df is None or len(df) == 0:
            return None

        cash = self.initial_capital
        holdings = []
        trades = []
        daily_values = []

        first_price = df.iloc[0]['close']
        first_shares = self.calculate_buy_shares(self.part_capital, first_price)

        if first_shares > 0:
            cost = first_shares * first_price
            fee = self.calculate_buy_fee(cost)
            total_cost = cost + fee

            if total_cost <= cash:
                cash -= total_cost
                holdings.append({
                    'price': first_price,
                    'shares': first_shares,
                    'cost': total_cost
                })
                trades.append({
                    'date': df.iloc[0]['date'],
                    'action': '买入',
                    'price': first_price,
                    'shares': first_shares,
                    'amount': cost,
                    'fee': fee,
                    'cash': cash,
                    'holdings': len(holdings),
                    'reason': '首次建仓'
                })

        for i in range(1, len(df)):
            current_price = df.iloc[i]['close']
            current_date = df.iloc[i]['date']

            if len(holdings) > 0:
                total_shares = sum(h['shares'] for h in holdings)
                avg_cost = sum(h['price'] * h['shares'] for h in holdings) / total_shares
            else:
                avg_cost = 0
                total_shares = 0

            # 卖出条件
            if len(holdings) > 0 and total_shares > 0:
                profit_pct = (current_price - avg_cost) / avg_cost

                if profit_pct >= y_percent / 100:
                    sell_holdings = holdings[0]
                    sell_shares = sell_holdings['shares']
                    sell_amount = sell_shares * current_price
                    fee = self.calculate_sell_fee(sell_amount)
                    net_amount = sell_amount - fee

                    cash += net_amount
                    holdings.pop(0)

                    trades.append({
                        'date': current_date,
                        'action': '卖出',
                        'price': current_price,
                        'shares': sell_shares,
                        'amount': sell_amount,
                        'fee': fee,
                        'cash': cash,
                        'holdings': len(holdings),
                        'reason': f'上涨{y_percent}%止盈'
                    })

            # 买入条件
            if len(holdings) < self.num_parts:
                if len(holdings) > 0:
                    base_price = holdings[-1]['price']
                else:
                    base_price = current_price

                decline_pct = (current_price - base_price) / base_price

                if decline_pct <= -x_percent / 100:
                    buy_capital = min(self.part_capital, cash)
                    buy_shares = self.calculate_buy_shares(buy_capital, current_price)

                    if buy_shares > 0:
                        cost = buy_shares * current_price
                        fee = self.calculate_buy_fee(cost)
                        total_cost = cost + fee

                        if total_cost <= cash:
                            cash -= total_cost
                            holdings.append({
                                'price': current_price,
                                'shares': buy_shares,
                                'cost': total_cost
                            })
                            trades.append({
                                'date': current_date,
                                'action': '买入',
                                'price': current_price,
                                'shares': buy_shares,
                                'amount': cost,
                                'fee': fee,
                                'cash': cash,
                                'holdings': len(holdings),
                                'reason': f'下跌{x_percent}%补仓'
                            })

            stock_value = sum(h['shares'] * current_price for h in holdings)
            total_value = cash + stock_value

            benchmark_shares = self.calculate_buy_shares(self.initial_capital, first_price)
            if benchmark_shares > 0:
                benchmark_cost = benchmark_shares * first_price
                benchmark_fee = self.calculate_buy_fee(benchmark_cost)
                benchmark_value = (self.initial_capital - benchmark_cost - benchmark_fee) + benchmark_shares * current_price
            else:
                benchmark_value = self.initial_capital

            daily_values.append({
                'date': current_date,
                'price': current_price,
                'cash': cash,
                'stock_value': stock_value,
                'total_value': total_value,
                'holdings_count': len(holdings),
                'total_shares': sum(h['shares'] for h in holdings),
                'avg_cost': avg_cost if len(holdings) > 0 else 0,
                'benchmark_value': benchmark_value
            })

        final_price = df.iloc[-1]['close']
        final_stock_value = sum(h['shares'] * final_price for h in holdings)
        final_total = cash + final_stock_value

        total_return = (final_total - self.initial_capital) / self.initial_capital * 100
        buy_hold_return = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100

        return {
            'trades': pd.DataFrame(trades),
            'daily_values': pd.DataFrame(daily_values),
            'final_cash': cash,
            'final_stock_value': final_stock_value,
            'final_total': final_total,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'num_trades': len(trades),
            'x': x_percent,
            'y': y_percent
        }

# ==================== 最优参数搜索 ====================

def find_optimal_params(df, strategy, target_return, x_range=(1, 20), y_range=(1, 30), step=0.5):
    """网格搜索最优参数"""
    results = []

    x_values = np.arange(x_range[0], x_range[1] + step, step)
    y_values = np.arange(y_range[0], y_range[1] + step, step)

    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(x_values) * len(y_values)
    count = 0

    for x in x_values:
        for y in y_values:
            count += 1
            progress = count / total
            progress_bar.progress(min(progress, 0.99))
            status_text.text(f"正在回测参数组合: 下跌{x}%买入, 上涨{y}%卖出... ({count}/{total})")

            result = strategy.backtest(df.copy(), x, y)
            if result and result['total_return'] >= target_return:
                daily_df = result['daily_values']
                if len(daily_df) > 0:
                    daily_returns = daily_df['total_value'].pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252) * 100
                    max_drawdown = ((daily_df['total_value'].cummax() - daily_df['total_value']) / 
                                   daily_df['total_value'].cummax()).max() * 100

                    results.append({
                        'x': x,
                        'y': y,
                        'return': result['total_return'],
                        'volatility': volatility,
                        'max_drawdown': max_drawdown,
                        'trades': result['num_trades'],
                        'sharpe': result['total_return'] / volatility if volatility > 0 else 0
                    })

    progress_bar.empty()
    status_text.empty()

    if not results:
        return None

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe', ascending=False).reset_index(drop=True)

    return results_df

# ==================== 可视化模块 ====================

def plot_results(df, result, stock_name):
    """绘制收益曲线和交易点"""
    daily_df = result['daily_values']
    trades_df = result['trades']

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f'{stock_name} 价格走势与交易点', '资产价值对比'),
        row_heights=[0.6, 0.4]
    )

    fig.add_trace(
        go.Scatter(x=daily_df['date'], y=daily_df['price'],
                  mode='lines', name='收盘价', line=dict(color='#1f77b4', width=1)),
        row=1, col=1
    )

    if not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == '买入']
        if not buy_trades.empty:
            fig.add_trace(
                go.Scatter(x=buy_trades['date'], y=buy_trades['price'],
                          mode='markers', name='买入',
                          marker=dict(color='red', size=10, symbol='triangle-up')),
                row=1, col=1
            )

        sell_trades = trades_df[trades_df['action'] == '卖出']
        if not sell_trades.empty:
            fig.add_trace(
                go.Scatter(x=sell_trades['date'], y=sell_trades['price'],
                          mode='markers', name='卖出',
                          marker=dict(color='green', size=10, symbol='triangle-down')),
                row=1, col=1
            )

    fig.add_trace(
        go.Scatter(x=daily_df['date'], y=daily_df['total_value'],
                  mode='lines', name='策略资产', line=dict(color='#ff7f0e', width=2)),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=daily_df['date'], y=daily_df['benchmark_value'],
                  mode='lines', name='买入持有', line=dict(color='gray', width=1, dash='dash')),
        row=2, col=1
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="资产价值(元)", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)

    return fig

def plot_heatmap(results_df):
    """绘制参数热力图"""
    pivot = results_df.pivot(index='x', columns='y', values='return')

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        colorbar=dict(title='收益率(%)'),
        hovertemplate='下跌买入: %{y}%<br>上涨卖出: %{x}%<br>收益率: %{z:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title="参数组合收益率热力图",
        xaxis_title="上涨卖出幅度 y(%)",
        yaxis_title="下跌买入幅度 x(%)",
        height=500,
        template='plotly_white'
    )

    return fig

# ==================== 主程序 ====================

def main():
    st.markdown('<div class="main-header">📈 智能波段交易模拟器</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ 参数配置")

        market_type = st.selectbox(
            "市场类型",
            ["A股", "可转债", "ETF基金", "场内基金"],
            help="选择要分析的市场类型"
        )

        if market_type == "A股":
            symbol = st.text_input("股票代码", "000001", 
                                  help="输入6位数字代码，如: 000001(平安银行)")
        elif market_type == "可转债":
            symbol = st.text_input("转债代码", "127045", 
                                  help="输入转债代码，如: 127045")
        else:
            symbol = st.text_input("基金代码", "510300", 
                                  help="输入基金代码，如: 510300(沪深300ETF)")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", datetime(2024, 1, 1))
        with col2:
            end_date = st.date_input("结束日期", datetime.now())

        st.subheader("投资参数")
        initial_capital = st.number_input("初始资金(元)", value=100000, step=10000)
        target_return = st.number_input("目标收益率(%)", value=10.0, step=1.0,
                                       help="期望达到的最低年化收益率")

        st.subheader("策略参数")
        x_min = st.number_input("最小下跌幅度(%)", value=1.0, step=0.5)
        x_max = st.number_input("最大下跌幅度(%)", value=15.0, step=0.5)
        y_min = st.number_input("最小上涨幅度(%)", value=1.0, step=0.5)
        y_max = st.number_input("最大上涨幅度(%)", value=20.0, step=0.5)
        step = st.number_input("搜索步长(%)", value=0.5, step=0.1, 
                              help="x和y的搜索精度")

        st.subheader("手续费设置")
        buy_fee = st.number_input("买入费率(%)", value=0.03, step=0.01,
                                 help="佣金，最低5元")
        sell_fee = st.number_input("卖出费率(%)", value=0.13, step=0.01,
                                  help="佣金0.03% + 印花税0.1%")

        run_button = st.button("🚀 开始回测", type="primary", use_container_width=True)

    if run_button:
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期！")
            return

        with st.spinner("正在获取数据..."):
            stock_name = get_stock_name(symbol, market_type)
            df = get_stock_data(symbol, start_date, end_date, market_type)

        if df is None or len(df) < 10:
            st.error("获取数据失败或数据不足，请检查代码和日期范围！")
            return

        st.success(f"✅ 成功获取 {stock_name}({symbol}) 的 {len(df)} 个交易日数据")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("起始价格", f"¥{df.iloc[0]['close']:.2f}")
        with col2:
            st.metric("结束价格", f"¥{df.iloc[-1]['close']:.2f}")
        with col3:
            change = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100
            st.metric("期间涨跌", f"{change:+.2f}%")
        with col4:
            st.metric("交易日数", len(df))

        strategy = TradingStrategy(
            initial_capital=initial_capital,
            buy_fee_rate=buy_fee/100,
            sell_fee_rate=sell_fee/100
        )

        with st.spinner("正在搜索最优参数组合，这可能需要几分钟..."):
            results_df = find_optimal_params(
                df, strategy, target_return,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                step=step
            )

        if results_df is None or results_df.empty:
            st.warning(f"⚠️ 未找到能达到目标收益率 {target_return}% 的参数组合，请放宽条件重试。")

            st.subheader("📊 所有参数组合收益分布")
            all_results = []
            x_values = np.arange(x_min, x_max + step, step)
            y_values = np.arange(y_min, y_max + step, step)

            for x in x_values:
                for y in y_values:
                    result = strategy.backtest(df.copy(), x, y)
                    if result:
                        all_results.append({
                            'x': x, 'y': y, 'return': result['total_return']
                        })

            if all_results:
                all_df = pd.DataFrame(all_results)
                fig = plot_heatmap(all_df)
                st.plotly_chart(fig, use_container_width=True)

            return

        st.subheader("🏆 最优参数组合")

        best = results_df.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("最佳下跌买入", f"{best['x']:.1f}%")
        with col2:
            st.metric("最佳上涨卖出", f"{best['y']:.1f}%")
        with col3:
            st.metric("预期收益率", f"{best['return']:.2f}%")
        with col4:
            st.metric("最大回撤", f"{best['max_drawdown']:.2f}%")
        with col5:
            st.metric("交易次数", int(best['trades']))

        best_result = strategy.backtest(df.copy(), best['x'], best['y'])

        st.subheader("📈 收益曲线与交易点")
        fig = plot_results(df, best_result, stock_name)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 参数组合收益率热力图")
        heatmap_fig = plot_heatmap(results_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.subheader("📋 交易明细")
        trades_df = best_result['trades']
        if not trades_df.empty:
            display_df = trades_df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['price'] = display_df['price'].apply(lambda x: f"¥{x:.2f}")
            display_df['amount'] = display_df['amount'].apply(lambda x: f"¥{x:,.2f}")
            display_df['fee'] = display_df['fee'].apply(lambda x: f"¥{x:.2f}")
            display_df['cash'] = display_df['cash'].apply(lambda x: f"¥{x:,.2f}")

            def color_action(val):
                if val == '买入':
                    return 'color: red; font-weight: bold'
                elif val == '卖出':
                    return 'color: green; font-weight: bold'
                return ''

            styled_df = display_df.style.applymap(color_action, subset=['action'])
            st.dataframe(styled_df, use_container_width=True, height=400)

            csv = trades_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下载交易明细(CSV)",
                data=csv,
                file_name=f"{symbol}_trades_{best['x']}_{best['y']}.csv",
                mime="text/csv"
            )

        st.subheader("📊 收益对比分析")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>策略收益</h4>
                <p style="font-size: 2rem; color: #1f77b4;">¥{:.2f}</p>
                <p>收益率: {:.2f}%</p>
                <p>交易次数: {}次</p>
            </div>
            """.format(best_result['final_total'], best_result['total_return'], 
                      best_result['num_trades']), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>买入持有收益</h4>
                <p style="font-size: 2rem; color: gray;">¥{:.2f}</p>
                <p>收益率: {:.2f}%</p>
                <p>交易次数: 1次</p>
            </div>
            """.format(
                initial_capital * (1 + best_result['buy_hold_return']/100),
                best_result['buy_hold_return']
            ), unsafe_allow_html=True)

        st.subheader("📋 前10个备选参数组合")
        top10 = results_df.head(10)[['x', 'y', 'return', 'max_drawdown', 'trades', 'sharpe']]
        top10.columns = ['下跌买入x(%)', '上涨卖出y(%)', '收益率(%)', '最大回撤(%)', '交易次数', '夏普比率']
        st.dataframe(top10, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            <strong>⚠️ 风险提示</strong><br>
            1. 本工具仅供学习研究，不构成投资建议<br>
            2. 历史回测结果不代表未来收益<br>
            3. 实际交易需考虑滑点、流动性等因素<br>
            4. 网格交易策略在单边行情中可能表现不佳
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
