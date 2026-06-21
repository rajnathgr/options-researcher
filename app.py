"""
Options Research Agent — Streamlit App
Converted from options_research.ipynb
"""

import os
import json
import math
import asyncio
import numpy as np
import streamlit as st
import yfinance as yf

from datetime import date, datetime
from scipy.stats import norm
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(override=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Research Agent",
    page_icon="📈",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark terminal feel — suits a data/trading tool */
    [data-testid="stAppViewContainer"] {
        background-color: #0f1117;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    /* Header */
    .app-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.25rem;
    }
    .app-title {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: -0.5px;
    }
    .app-sub {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem;
    }
    /* Input box */
    [data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.85rem !important;
    }
    /* Sidebar labels */
    .sidebar-label {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.25rem;
    }
    .sidebar-value {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.85rem;
        color: #58a6ff;
    }
    .divider {
        border-top: 1px solid #21262d;
        margin: 1rem 0;
    }
    /* Suggestion chips */
    .stButton > button {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #8b949e !important;
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.72rem !important;
        border-radius: 6px !important;
        padding: 0.3rem 0.6rem !important;
        width: 100%;
        text-align: left !important;
    }
    .stButton > button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Strategy constants ────────────────────────────────────────────────────────
MIN_DTE        = 0
MAX_DTE        = 14
RISK_FREE_RATE = 0.045


# ── Black-Scholes put delta ───────────────────────────────────────────────────
def calculate_put_delta(S: float, K: float, T: float, iv: float):
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        return round(abs(norm.cdf(d1) - 1), 4)
    except Exception:
        return None


# ── Tools ─────────────────────────────────────────────────────────────────────
@function_tool
def get_options_chain(ticker: str) -> str:
    """
    Fetches put options data for a given stock ticker from Yahoo Finance.
    Returns options expiring between 0 and 14 days from today, including
    strike, bid, ask, mid, effective premium, implied volatility, delta
    (probability ITM), volume, and open interest.
    """
    today  = date.today()
    result = {"ticker": ticker.upper(), "stock_price": None, "puts": [], "error": None, "note": None}

    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            result["error"] = f"Could not retrieve price history for {ticker}"
            return json.dumps(result)

        stock_price           = round(float(hist["Close"].iloc[-1]), 2)
        result["stock_price"] = stock_price

        expiries = t.options
        if not expiries:
            result["error"] = f"No options found for {ticker}"
            return json.dumps(result)

        valid_expiries = []
        for exp in expiries:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte      = (exp_date - today).days
            if MIN_DTE <= dte <= MAX_DTE:
                valid_expiries.append((exp, dte))

        if not valid_expiries:
            result["error"] = (
                f"No expirations in the {MIN_DTE}–{MAX_DTE} DTE window for {ticker}. "
                f"Available expirations: {list(expiries[:6])}"
            )
            return json.dumps(result)

        puts      = []
        using_mid = False

        for exp, dte in valid_expiries:
            chain = t.option_chain(exp)
            df    = chain.puts
            if df.empty:
                continue

            T = dte / 365.0

            for _, row in df.iterrows():
                bid    = float(row.get("bid", 0) or 0)
                ask    = float(row.get("ask", 0) or 0)
                vol    = row.get("volume", 0)
                oi     = row.get("openInterest", 0)
                iv     = float(row.get("impliedVolatility", 0) or 0)
                strike = float(row["strike"])

                mid               = round((bid + ask) / 2, 2) if ask > 0 else 0
                effective_premium = bid if bid > 0 else mid
                if bid == 0 and mid > 0:
                    using_mid = True

                delta = calculate_put_delta(S=stock_price, K=strike, T=T, iv=iv)

                premium_per_contract = round(effective_premium * 100, 2)
                capital_required     = round(strike * 100, 2)
                premium_on_5k        = (
                    round((premium_per_contract / capital_required) * 5000, 2)
                    if capital_required > 0 else 0
                )

                puts.append({
                    "strike":               strike,
                    "expiry":               exp,
                    "dte":                  dte,
                    "bid":                  bid,
                    "ask":                  ask,
                    "mid":                  mid,
                    "effective_premium":    effective_premium,
                    "impliedVolatility":    round(iv, 4),
                    "delta":                delta,
                    "prob_itm_pct":         round(delta * 100, 1) if delta else None,
                    "volume":               int(vol) if vol and not math.isnan(float(vol)) else 0,
                    "openInterest":         int(oi)  if oi  and not math.isnan(float(oi))  else 0,
                    "inTheMoney":           bool(row.get("inTheMoney", False)),
                    "premium_per_contract": premium_per_contract,
                    "capital_required":     capital_required,
                    "premium_on_5k":        premium_on_5k,
                })

        result["puts"] = puts
        if using_mid:
            result["note"] = (
                "Market is closed — bid prices are 0. "
                "effective_premium uses midpoint (bid+ask)/2 as an estimate."
            )

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


@function_tool
def get_analyst_recommendations(ticker: str) -> str:
    """
    Fetches the latest analyst recommendations for a given stock ticker
    from Yahoo Finance.
    """
    result = {"ticker": ticker.upper(), "summary": None, "recent": [], "error": None}

    try:
        t       = yf.Ticker(ticker)
        summary = t.recommendations_summary
        if summary is not None and not summary.empty:
            result["summary"] = summary.to_dict(orient="records")

        recs = t.recommendations
        if recs is not None and not recs.empty:
            recent       = recs.tail(10).copy()
            recent.index = recent.index.astype(str)
            result["recent"] = recent.reset_index().to_dict(orient="records")

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


@function_tool
def get_company_info(ticker: str) -> str:
    """
    Fetches company profile information for a given stock ticker from Yahoo Finance.
    """
    result = {"ticker": ticker.upper(), "info": None, "error": None}

    try:
        t    = yf.Ticker(ticker)
        info = t.info

        result["info"] = {
            "name":                info.get("longName"),
            "description":         info.get("longBusinessSummary"),
            "sector":              info.get("sector"),
            "industry":            info.get("industry"),
            "market_cap":          info.get("marketCap"),
            "beta":                info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low":  info.get("fiftyTwoWeekLow"),
            "average_volume":      info.get("averageVolume"),
            "employees":           info.get("fullTimeEmployees"),
            "website":             info.get("website"),
        }

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


@function_tool
def get_news(ticker: str) -> str:
    """
    Fetches the most recent news headlines for a given stock ticker from Yahoo Finance.
    """
    result = {"ticker": ticker.upper(), "articles": [], "error": None}

    try:
        t    = yf.Ticker(ticker)
        news = t.news

        if not news:
            result["error"] = f"No news found for {ticker}"
            return json.dumps(result)

        articles = []
        for item in news[:10]:
            pub_time = item.get("providerPublishTime")
            pub_date = (
                datetime.utcfromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M UTC")
                if pub_time else None
            )
            articles.append({
                "headline":  item.get("title"),
                "publisher": item.get("publisher"),
                "published": pub_date,
                "url":       item.get("link"),
            })

        result["articles"] = articles

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


@function_tool
def get_stock_price(ticker: str) -> str:
    """
    Fetches the current stock price for a given ticker from Yahoo Finance.
    """
    result = {"ticker": ticker.upper(), "price": None, "error": None}

    try:
        t    = yf.Ticker(ticker)
        info = t.info

        result["price"] = {
            "current":        info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "day_high":       info.get("dayHigh"),
            "day_low":        info.get("dayLow"),
            "volume":         info.get("volume"),
            "currency":       info.get("currency", "USD"),
        }

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result)


# ── Agent (cached — only created once per session) ────────────────────────────
@st.cache_resource
def build_agent():
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

    claude_client = AsyncOpenAI(
        base_url="https://api.anthropic.com/v1/",
        api_key=api_key,
    )

    system_prompt = f"""You are an options analyst assistant for a retail trader who sells cash-secured puts.

TOOLS AVAILABLE:
- get_options_chain: fetches put options expiring in 0–14 days for a ticker
- get_analyst_recommendations: fetches analyst ratings and recent actions for a ticker
- get_company_info: fetches company profile — description, sector, industry, market cap, beta, 52-week range, average volume
- get_news: fetches the 10 most recent news headlines for a ticker
- get_stock_price: fetches current price, previous close, day high/low, and volume for a ticker

WHEN TO CALL EACH TOOL:
- User asks for options data → get_options_chain
- User asks for analyst ratings or sentiment → get_analyst_recommendations
- User asks about the company, what it does, its profile → get_company_info
- User asks for news or what's moving the stock → get_news
- User asks for a "full picture" or "full analysis" → call all four tools

STRATEGY CONTEXT:
- Trader sells cash-secured puts only
- DTE window: {MIN_DTE} to {MAX_DTE} days
- Max delta threshold: 0.30 (~30% probability of expiring in the money)
- Capital per trade: $5,000
- Minimum premium target: $70 per $5,000 deployed
- The tool pre-calculates premium_on_5k for every contract — use it directly

DELTA / PROBABILITY ITM:
The 'delta' field is the absolute value of the Black-Scholes put delta.
'prob_itm_pct' is delta expressed as a percentage.
A delta of 0.28 means ~28% chance of expiring in the money.
Flag contracts where delta > 0.30 as outside the trader's risk threshold.

PRICING NOTE:
When the market is closed, bid prices show as 0.
In that case effective_premium uses midpoint (bid+ask)/2.
Always flag this clearly when it applies.

OUTPUT FORMAT:

For options data present a clean table:
Strike | Expiry | DTE | Bid | Ask | Mid | IV% | Delta | Prob ITM% | Vol | OI | On $5k

Sort by strike ascending. After the table:
- Note the current stock price
- Note total contracts found in the DTE window
- Highlight contracts that meet ALL strategy criteria (delta ≤ 0.30 AND premium_on_5k ≥ $70)
- Flag if prices are estimated (mid) vs live (bid)

For analyst recommendations show:
- Summary table: Strong Buy | Buy | Hold | Sell | Strong Sell counts AND percentage of total
  e.g. Strong Buy: 5 (22.7%) | Buy: 10 (45.5%) etc.
- 10 most recent analyst actions: Date | Firm | From | To
"""

    return Agent(
        name="Options Analyst Agent",
        instructions=system_prompt,
        model=OpenAIChatCompletionsModel(
            model="claude-sonnet-4-6",
            openai_client=claude_client,
        ),
        tools=[
            get_options_chain,
            get_analyst_recommendations,
            get_company_info,
            get_news,
            get_stock_price,
        ],
    )


# ── Async runner helper ───────────────────────────────────────────────────────
async def run_agent(agent, full_input: str) -> str:
    result = await Runner.run(agent, input=full_input)
    return result.final_output


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-sub">Strategy Config</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">DTE Window</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">{MIN_DTE} – {MAX_DTE} days</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top:0.75rem">Max Delta</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">0.30</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top:0.75rem">Capital / Trade</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">$5,000</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top:0.75rem">Min Premium on $5k</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">$70</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top:0.75rem">Risk-Free Rate</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-value">{RISK_FREE_RATE*100:.1f}%</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Quick Prompts</div>', unsafe_allow_html=True)

    suggestions = [
        "Full analysis on IONQ",
        "Options chain for OKLO",
        "What's moving IREN today?",
        "Analyst ratings for HUT",
        "Stock price for NVDA",
    ]

    for s in suggestions:
        if st.button(s, key=f"btn_{s}"):
            st.session_state["pending_prompt"] = s

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("🗑 Clear conversation", key="clear"):
        st.session_state.messages = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span class="app-title">📈 Options Research</span>
</div>
<div class="app-sub" style="margin-bottom:1.5rem">Cash-Secured Puts · 0–14 DTE · Powered by Claude</div>
""", unsafe_allow_html=True)


# ── Render existing chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Handle sidebar quick-prompt injection ─────────────────────────────────────
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = build_agent()

    full_input = ""
    for m in st.session_state.messages[:-1]:
        prefix = "User" if m["role"] == "user" else "Assistant"
        full_input += f"{prefix}: {m['content']}\n"
    full_input += f"User: {prompt}"

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            response = asyncio.run(run_agent(agent, full_input))
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


# ── Main chat input ───────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about a ticker — options chain, news, analyst ratings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = build_agent()

    full_input = ""
    for m in st.session_state.messages[:-1]:
        prefix = "User" if m["role"] == "user" else "Assistant"
        full_input += f"{prefix}: {m['content']}\n"
    full_input += f"User: {prompt}"

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            response = asyncio.run(run_agent(agent, full_input))
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
