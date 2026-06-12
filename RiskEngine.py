import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

ROLLING_WINDOW = 30

st.set_page_config(page_title="Portfolio Risk Intelligence",
                   page_icon="⚡",
                   layout="wide")

st.title("🐦‍🔥 Portfolio Risk Intelligence Dashboard 🐦‍🔥")

uploaded_file = st.file_uploader(
    "Upload Portfolio (Ticker, Weight)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    portfolio = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    mapping = pd.read_csv("ind_nifty500list.csv")[
        ["Ticker", "ISIN Code", "Company Name", "Industry"]
    ]

    portfolio = portfolio.merge(mapping, on="Ticker", how="left")

    portfolio = portfolio[[
        "ISIN Code","Ticker","Company Name","Industry","Weight"
    ]]

    st.session_state["portfolio"] = portfolio

if "portfolio" in st.session_state:

    portfolio = st.session_state["portfolio"]

    st.subheader("Portfolio Holdings")
    st.dataframe(portfolio, use_container_width=True, hide_index=True)

    total_weight = portfolio["Weight"].sum()

    if abs(total_weight - 100) < 0.01:
        st.success(f"Portfolio weights sum to {total_weight:.2f}%")
    else:
        st.warning(f"Portfolio weights sum to {total_weight:.2f}%")

    lookback_years = st.number_input(
        "Lookback Years",
        min_value=1,
        max_value=20,
        value=10
    )

    if st.button("Download Price Data"):

        end_date = datetime.today()
        start_date = end_date - relativedelta(years=lookback_years)

        tickers = portfolio["Ticker"].dropna().unique().tolist()

        stock_prices = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )["Close"]

        if isinstance(stock_prices, pd.Series):
            stock_prices = stock_prices.to_frame()

        nifty_prices = yf.download(
            "^NSEI",
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )["Close"]

        
        availability = pd.DataFrame(index=stock_prices.columns)

        availability["First Available Date"] = [
            stock_prices[c].first_valid_index()
            for c in stock_prices.columns
        ]

        availability["Last Available Date"] = [
            stock_prices[c].last_valid_index()
            for c in stock_prices.columns
        ]

        availability["Years Available"] = (
            (
                availability["Last Available Date"]
                - availability["First Available Date"]
            ).dt.days / 365.25
        ).round(1)

        availability["First Available Date"] = (
            availability["First Available Date"]
            .dt.strftime("%d-%m-%Y")
        )

        availability["Last Available Date"] = (
            availability["Last Available Date"]
            .dt.strftime("%d-%m-%Y")
        )

        st.session_state["prices"] = stock_prices
        st.session_state["nifty"] = nifty_prices
        st.session_state["availability"] = availability
        st.session_state["lookback"] = lookback_years

    
if "prices" in st.session_state:

    portfolio = st.session_state["portfolio"]
    prices = st.session_state["prices"]
    nifty_prices = st.session_state["nifty"]
    availability = st.session_state["availability"]

    st.subheader("Data Availability")
    st.dataframe(availability, use_container_width=True)
    insufficient_history = availability[
    availability["Years Available"] < lookback_years
    ]

    if len(insufficient_history) == 0:

        st.success(
            f"✅ All stocks have the full {lookback_years}-year history."
        )

    else:

        st.warning(
            f"⚠️ {len(insufficient_history)} stock(s) do not have the full {lookback_years}-year history."
        )

        for ticker, row in insufficient_history.iterrows():

            st.write(
                f"• {ticker}: Only {row['Years Available']:.1f} years available "
                f"(data starts from {row['First Available Date']})"
            )
            

    stock_returns = prices.pct_change().dropna()
    nifty_returns = nifty_prices.pct_change().dropna()

    weights = portfolio.set_index("Ticker")["Weight"] / 100

    common = stock_returns.columns.intersection(weights.index)

    stock_returns = stock_returns[common]
    weights = weights[common]

    portfolio_returns = stock_returns.mul(weights, axis=1).sum(axis=1)
    
    st.subheader("📈 Portfolio vs Nifty Performance")

    comparison_df = pd.concat(
        [portfolio_returns, nifty_returns],
        axis=1
    ).dropna()

    comparison_df.columns = [
        "Portfolio",
        "Nifty"
    ]

    growth_df = (
        1 + comparison_df
    ).cumprod() * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=growth_df.index,
            y=growth_df["Portfolio"],
            mode="lines",
            name="Portfolio"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=growth_df.index,
            y=growth_df["Nifty"],
            mode="lines",
            name="Nifty 50"
        )
    )

    fig.update_layout(
        title="Growth of ₹100 Invested",
        template="plotly_dark",
        hovermode="x unified",
        height=550,
        yaxis_title="Portfolio Value (₹)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    correlation_matrix = stock_returns.corr()

    beta_dict = {}

    for ticker in stock_returns.columns:

        merged = pd.concat(
            [stock_returns[ticker], nifty_returns],
            axis=1
        ).dropna()

        merged.columns = ["stock", "nifty"]

        beta_dict[ticker] = (
            merged["stock"].cov(merged["nifty"])
            / merged["nifty"].var()
        )

    beta_df = pd.DataFrame(beta_dict.items(), columns=["Ticker", "Beta"])

    weights = (
        portfolio
        .set_index("Ticker")["Weight"]
        / 100
    )

    beta_df["Weight"] = (
        beta_df["Ticker"]
        .map(weights)
    )

    # beta_df["Risk Contribution"] = (
    #     beta_df["Weight"]
    #     * beta_df["Beta"]
    # )

    beta_df["Risk Contribution %"] = (
        (beta_df["Weight"] * beta_df["Beta"])
        / (beta_df["Weight"] * beta_df["Beta"]).sum()
        * 100
    )

    portfolio_beta = (
        beta_df.set_index("Ticker")
        .loc[weights.index]["Beta"]
        .mul(weights)
        .sum()
    )
    st.subheader("Individual Stock Betas & Risk Contribution")

    st.dataframe(
        beta_df.sort_values(
            "Beta",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )
    

    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

    cumulative_return = (1 + portfolio_returns).prod()

    years = (
        portfolio_returns.index[-1]
        - portfolio_returns.index[0]
    ).days / 365.25

    portfolio_cagr = cumulative_return ** (1 / years) - 1

    portfolio_sharpe = (
        portfolio_returns.mean() / portfolio_returns.std()
    ) * np.sqrt(252)
    
    nifty_cagr = (1 + nifty_returns).prod() ** (1 / years) - 1
    nifty_cagr = float(nifty_cagr)
    
    outperformance = portfolio_cagr - nifty_cagr
    
    aligned = pd.concat(
    [portfolio_returns, nifty_returns],
        axis=1
    ).dropna()

    aligned.columns = [
        "Portfolio",
        "Nifty"
    ]
    
    active_returns = (
    aligned["Portfolio"]
    - aligned["Nifty"]
    )

    tracking_error = (
        active_returns.std()
        * np.sqrt(252)
    )

    information_ratio = (
        active_returns.mean()
        * 252
    ) / tracking_error
    

    st.subheader("Risk Summary")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
    c2.metric("Volatility", f"{portfolio_volatility:.2%}")
    c3.metric("CAGR", f"{portfolio_cagr:.2%}")
    c4.metric("Sharpe", f"{portfolio_sharpe:.2f}")
    c5.metric("Outperformance", f"{outperformance   :.2%}")
    c6.metric("Tracking Error", f"{tracking_error:.2%}")
    c7.metric("Information Ratio", f"{information_ratio:.2f}")
    

    st.subheader("Individual Stock Betas")
    st.dataframe(beta_df.sort_values("Beta", ascending=False),
                 use_container_width=True,
                 hide_index=True)

    st.subheader("Correlation Heatmap")

    fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale="darkmint",
                zmin=-1,
                zmax=1
            )
        )

    fig.update_layout(
            height=400,
            template="plotly_dark",
            title="Stock Return Correlation Heatmap",
        )

    st.plotly_chart(
            fig,
            use_container_width=True
        )

    

    st.subheader("Rolling Metrics")
        # ==========================================
    # 30-DAY ROLLING METRICS
    # ==========================================

    aligned = pd.concat(
        [portfolio_returns, nifty_returns],
        axis=1
    ).dropna()

    aligned.columns = [
        "Portfolio",
        "Nifty"
    ]

    rolling_sharpe = (
        aligned["Portfolio"]
        .rolling(30)
        .mean()
        /
        aligned["Portfolio"]
        .rolling(30)
        .std()
    ) * np.sqrt(252)
    
    active_returns = (aligned["Portfolio"] - aligned["Nifty"])
    rolling_tracking_error = (
    active_returns
    .rolling(30)
    .std()
    * np.sqrt(252)
    )

    rolling_information_ratio = (
        active_returns
        .rolling(30)
        .mean()
        * 252
    ) / rolling_tracking_error

    downside_returns = aligned["Portfolio"].copy()

    downside_returns[
        downside_returns > 0
    ] = 0

    rolling_downside_std = (
        downside_returns
        .rolling(30)
        .std()
    )

    rolling_sortino = (
        aligned["Portfolio"]
        .rolling(30)
        .mean()
        /
        rolling_downside_std
    ) * np.sqrt(252)


    # Remove NaNs generated by rolling window


    rolling_sharpe = rolling_sharpe.dropna()
    rolling_sortino = rolling_sortino.dropna()
    rolling_information_ratio = rolling_information_ratio.dropna()
    rolling_tracking_error = rolling_tracking_error.dropna()
    # st.write(rolling_information_ratio.describe())

    # ==========================================
    # ROLLING RISK CHART
    # ==========================================

    # st.subheader("30-Day Rolling Risk Metrics")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode="lines",
            name="Rolling Sharpe",
            # fill="tozeroy",
            line=dict(color="cyan")
        )
    )


    fig.add_trace(
        go.Scatter(
            x=rolling_sortino.index,
            y=rolling_sortino,
            mode="lines",
            name="Rolling Sortino",
            line=dict(color="magenta")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=rolling_information_ratio.index,
            y=rolling_information_ratio,
            mode="lines",
            name="Rolling Information Ratio",
            line=dict(color="yellow")
        )
    )
    

    fig.update_layout(
        title="30-Day Rolling Sharpe,Sortino,Information Ratio",
        template="plotly_dark",
        hovermode="x unified",
        height=600,
        legend=dict(
            orientation="h",
            y=1.02
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # 
    #                        Phase - 2: Risk Engine
    # 
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        margin:40px 0;
    ">
        <div style="flex:1;height:1px;background:#444;"></div>
        <div style="
            padding:0 20px;
            color:#E6C068;
            font-family:Georgia, serif;
            font-style:italic;
            font-weight:bold;
            font-size:20px;">
            Downside Risk Metrics 
        </div>
        <div style="flex:1;height:1px;background:#444;"></div>
    </div>
    """,
    unsafe_allow_html=True
)
    
    st.subheader("📌 Downside Risk Metrics")
    
    # Max Drawdown
    cumulative_returns = (
        1 + portfolio_returns
    ).cumprod()

    running_peak = (
        cumulative_returns
        .cummax()
    )

    drawdown = (
        cumulative_returns
        / running_peak
        - 1
    )

    max_drawdown = drawdown.min()
    
    # Portfolio Drawdown Curve
    

    
    
    var_95 = np.percentile(
    portfolio_returns,
    5
    )
    cvar_95 = portfolio_returns[
    portfolio_returns <= var_95
    ].mean()
    var_99 = np.percentile(
    portfolio_returns,
    1
    )
    cvar_99 = portfolio_returns[
    portfolio_returns <= var_99
    ].mean()
    
    c1, c2, c3,c4 ,c5 = st.columns(5)

    c1.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2%}"
    )

    c2.metric(
        "VaR (95%)",
        f"{var_95:.2%}"
    )

    c3.metric(
        "CVaR (95%)",
        f"{cvar_95:.2%}"
    )
    c4.metric(
        "VaR (99%)",
        f"{var_99:.2%}"
    )

    c5.metric(
        "CVaR (99%)",
        f"{cvar_99:.2%}"
    )
    
    fig = go.Figure()

    # st.subheader("Portfolio Drawdown Curve")
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill="tozeroy",
            name="Drawdown"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Portfolio Drawdown History",
        yaxis_title="Drawdown",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    # Stress Testing - Hypothetical Scenarios
    st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        margin:50px 0;
    ">
        <div style="flex:1;height:1px;background:#444;"></div>
        <div style="
            padding:0 20px;
            color:#E6C068;
            font-family:Georgia, serif;
            font-style:italic;
            font-weight:bold;
            font-size:20px;">
            Stress Testing - Hypothetical Scenarios
        </div>
        <div style="flex:1;height:1px;background:#444;"></div>
    </div>
    """,
    unsafe_allow_html=True
    )
    
    stress_df = pd.DataFrame({
    "Scenario": [
        "Mild Correction",
        "Market Selloff",
        "Bear Market",
        "Severe Bear Market",
        "Market Crash"
    ],
    "Nifty Shock (%)": [
        -10,
        -20,
        -30,
        -40,
        -50
    ]
    })

    stress_df["Portfolio Impact (%)"] = (
        stress_df["Nifty Shock (%)"]
        * portfolio_beta
    )

    st.subheader("📌 Broader Market Shock Analysis")

    st.dataframe(
        stress_df,
        use_container_width=True,
        hide_index=True
    )
    

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=stress_df["Scenario"],
            y=stress_df["Portfolio Impact (%)"],
            text=stress_df["Portfolio Impact (%)"].round(1),
            textposition="outside"
        )
    )

    fig.update_layout(
        title="Estimated Portfolio Loss Under Market Stress",
        template="plotly_dark",
        yaxis_title="Portfolio Impact (%)",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
        
    st.subheader("📌 Sector Shock Analysis")
    
    sector_exposure = (
    portfolio
    .groupby("Industry")["Weight"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    )
    sector = st.selectbox(
    "Select Sector",
    sector_exposure["Industry"]
    )

    shock = st.slider(
        "Sector Shock (%)",
        -50,
        0,
        -20
    )
    
    sector_weight = (
    sector_exposure.loc[
        sector_exposure["Industry"] == sector,
        "Weight"
        ].iloc[0]
    )

    portfolio_impact = (
        sector_weight / 100
    ) * shock
            
    st.metric(
        f"Estimated Portfolio Impact from {shock}% shock to {sector} sector",
        f"{portfolio_impact:.2f}%"
    )
    
    
    st.subheader("📌 Historical Shock Events Analysis")
    
    stress_periods = {
    "Global Financial Crisis (2008)": (
        "2008-09-01",
        "2009-03-31"
    ),
    "COVID Crash (2020)": (
        "2020-02-01",
        "2020-03-31"
    ),
    "2022 Bear Market": (
        "2022-01-01",
        "2022-06-30"
    ),
    "Adani-Hindenburg Crisis": (
        "2023-01-24",
        "2023-02-28"
    )
    }
    results = []

    for scenario, (start, end) in stress_periods.items():

        portfolio_window = portfolio_returns.loc[start:end]
        nifty_window = nifty_returns.loc[start:end]

        if len(portfolio_window) == 0 or len(nifty_window) == 0:

            results.append([
                scenario,
                np.nan,
                np.nan
            ])

            continue

        portfolio_return = (
            (1 + portfolio_window).prod() - 1
        )

        nifty_return = (
            (1 + nifty_window).prod() - 1
        )

        if isinstance(portfolio_return, pd.Series):
            portfolio_return = portfolio_return.iloc[0]

        if isinstance(nifty_return, pd.Series):
            nifty_return = nifty_return.iloc[0]

        results.append([
            scenario,
            nifty_return,
            portfolio_return
        ])

    stress_df = pd.DataFrame(
        results,
        columns=[
            "Scenario",
            "Nifty Return",
            "Portfolio Return"
        ]
    )

    # st.subheader("Historical Stress Testing")

    stress_display = stress_df.copy()

    for col in ["Nifty Return", "Portfolio Return"]:

        stress_display[col] = stress_display[col].apply(
            lambda x: (
                f"{x:.2%}"
                if pd.notna(x)
                else "Insufficient History"
            )
        )

    st.dataframe(
        stress_display,
        use_container_width=True,
        hide_index=True
    )

