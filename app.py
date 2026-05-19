import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np

st.set_page_config(page_title="股票高抛低吸模拟器", layout="wide")
st.title("📈 股票三分之一高抛低吸模拟分析软件")
st.subheader("支持：股票 | 可转债 | ETF基金 | 自动寻找最优x/y")

# ========== 1. 输入参数 ==========
col1, col2 = st.columns(2)
with col1:
    code = st.text_input("证券代码（如 510300）", "510300")
    start_date = st.date_input("开始日期", value=pd.to_datetime("2023-01-01"))
with col2:
    end_date = st.date_input("结束日期", value=pd.to_datetime("2024-01-01"))
    target_profit = st.number_input("目标盈利率（%）", min_value=1.0, max_value=100.0, value=10.0)

start_date = str(start_date).replace("-", "")
end_date = str(end_date).replace("-", "")

# ========== 2. 获取数据 ==========
@st.cache_data(ttl=3600)
def get_data(code, start, end):
    try:
        df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start, end_date=end)
        df = df[["日期", "开盘", "收盘", "最高", "最低"]].copy()
        df.columns = ["date", "open", "close", "high", "low"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except:
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
            df = df[["日期", "开盘", "收盘", "最高", "最低"]].copy()
            df.columns = ["date", "open", "close", "high", "low"]
            df["date"] = pd.to_datetime(df["date"])
            return df
        except:
            return None

if st.button("🚀 开始运行回测"):
    df = get_data(code, start_date, end_date)
    if df is None or len(df) < 5:
        st.error("无法获取数据，请检查代码或日期")
        st.stop()

    st.success(f"✅ 成功获取 {len(df)} 天历史数据")
    st.dataframe(df.head(10), use_container_width=True)

    # ========== 3. 策略回测函数 ==========
    def run_strategy(x_pct, y_pct):
        cash = 100000.0
        part = cash / 3
        shares = 0.0
        trades = []
        last_buy_price = None

        # 第一天买入第一份
        first_price = df.iloc[0]["close"]
        buy_shares = part / first_price
        shares += buy_shares
        cash -= part
        last_buy_price = first_price
        trades.append({
            "date": df.iloc[0]["date"],
            "price": first_price,
            "action": "买入1/3",
            "shares": round(buy_shares, 2),
            "cash": round(cash, 2),
            "hold_shares": round(shares, 2)
        })

        for i in range(1, len(df)):
            price = df.iloc[i]["close"]
            dt = df.iloc[i]["date"]

            # 下跌买入 x%
            if last_buy_price is not None and cash >= part - 1:
                drop_rate = (last_buy_price - price) / last_buy_price * 100
                if drop_rate >= x_pct:
                    buy_shares = part / price
                    shares += buy_shares
                    cash -= part
                    last_buy_price = price
                    trades.append({
                        "date": dt, "price": price, "action": f"下跌{x_pct}%加仓",
                        "shares": round(buy_shares,2), "cash": round(cash,2), "hold_shares": round(shares,2)
                    })

            # 上涨卖出 y%
            if last_buy_price is not None and shares > 0:
                rise_rate = (price - last_buy_price) / last_buy_price * 100
                if rise_rate >= y_pct:
                    sell_shares = shares / 3
                    cash += sell_shares * price
                    shares -= sell_shares
                    trades.append({
                        "date": dt, "price": price, "action": f"上涨{y_pct}%减仓",
                        "shares": round(sell_shares,2), "cash": round(cash,2), "hold_shares": round(shares,2)
                    })

        # 最后清仓
        if shares > 0:
            final_price = df.iloc[-1]["close"]
            cash += shares * final_price
            trades.append({
                "date": df.iloc[-1]["date"], "price": final_price, "action": "回测结束清仓",
                "shares": round(shares,2), "cash": round(cash,2), "hold_shares": 0
            })

        final_asset = cash
        profit = final_asset - 100000
        profit_rate = profit / 100000 * 100
        return final_asset, profit, profit_rate, trades

    # ========== 4. 自动寻找最优 x / y ==========
    st.info("🔍 正在寻找满足目标收益的最优 x、y 参数...")
    best_x = None
    best_y = None
    best_diff = 999
    best_result = None
    best_trades = None

    # 遍历范围 1%~20%
    for x in np.arange(1, 20.5, 1):
        for y in np.arange(1, 20.5, 1):
            asset, profit, rate, trades = run_strategy(x, y)
            diff = abs(rate - target_profit)
            if diff < best_diff:
                best_diff = diff
                best_x = x
                best_y = y
                best_result = (asset, profit, rate)
                best_trades = trades

    asset, profit, rate = best_result
    st.subheader("📊 最优参数与回测结果")
    colA, colB, colC = st.columns(3)
    colA.metric("最优下跌买入 x", f"{best_x:.1f}%")
    colB.metric("最优上涨卖出 y", f"{best_y:.1f}%")
    colC.metric("目标收益率", f"{target_profit:.1f}%")

    col1, col2, col3 = st.columns(3)
    col1.metric("最终资产", f"{asset:.2f} 元")
    col2.metric("总收益", f"{profit:.2f} 元")
    col3.metric("实际收益率", f"{rate:.2f}%")

    st.subheader("📄 交易明细")
    trade_df = pd.DataFrame(best_trades)
    st.dataframe(trade_df, use_container_width=True, height=400)
