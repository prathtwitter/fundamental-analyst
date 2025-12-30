import streamlit as st
import yfinance as yf
import google.generativeai as genai
from duckduckgo_search import DDGS
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Company Analyzer | Elite Intelligence",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LUXURY STYLING (CSS) ---
st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #d4af37 !important; /* Gold */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
    }
    .stMetricValue {
        color: #d4af37 !important;
    }
    /* Cards/Containers */
    .css-1r6slb0 {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
    }
    /* Buttons */
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #388bfd;
    }
    /* Custom Status Indicators */
    .status-pass { color: #2ea043; font-weight: bold; }
    .status-fail { color: #da3633; font-weight: bold; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        color: #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #161b22;
        color: #d4af37;
        border-bottom: 2px solid #d4af37;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # 1. API Key Logic (Auto-load from Secrets or Manual Input)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("🔒 API Key Loaded Securely")
    else:
        api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key")
    
    st.markdown("---")
    region = st.selectbox("Select Region", ["USA 🇺🇸", "India 🇮🇳", "Canada 🇨🇦"])
    
    # Suffix logic for yfinance
    suffix_map = {"USA 🇺🇸": "", "India 🇮🇳": ".NS", "Canada 🇨🇦": ".TO"}
    suffix = suffix_map[region]
    
    st.info("💡 **Tip:**\nUSA: AAPL, MSFT\nIndia: RELIANCE, TCS\nCanada: SHOP, RY")

# --- MAIN HEADER ---
st.title("♟️ Company Analyzer")
st.markdown("*Strategic Fundamental & Technical Deep Dive powered by AI*")
st.markdown("---")

# --- FVG ENGINE (Strategic Sniper Logic) ---
class FVG_Engine:
    @staticmethod
    def resample_data(df, interval):
        """Resamples basic data into custom timeframes (4H, 6M, 12M)."""
        if df is None or df.empty: return None
        
        logic = {
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }
        try:
            # Drop NaN rows to prevent calculation errors
            resampled = df.resample(interval).agg(logic).dropna()
            return resampled
        except Exception as e:
            return None

    @staticmethod
    def find_bullish_fvgs(df, strict_mitigation=True):
        """
        Identifies Bullish FVGs (Candle 1 High < Candle 3 Low).
        Strict Logic: Invalid if ANY future candle wicks into the gap.
        """
        fvgs = []
        if df is None or len(df) < 3: return fvgs

        # Iterate through candles (leaving space for 3-candle pattern)
        for i in range(len(df) - 2):
            c1 = df.iloc[i]
            c2 = df.iloc[i+1] # The big move candle
            c3 = df.iloc[i+2]
            
            # FVG Condition: Gap between C1 High and C3 Low
            # Also ensure C2 is bullish (Close > Open) for valid context
            if c3['Low'] > c1['High'] and c2['Close'] > c2['Open']:
                top = c3['Low']
                bottom = c1['High']
                avg_price = (top + bottom) / 2
                
                is_valid = True
                if strict_mitigation:
                    # Check ALL subsequent candles for mitigation
                    future_candles = df.iloc[i+3:]
                    if not future_candles.empty:
                        min_low = future_candles['Low'].min()
                        # If any future Low goes below the Top of the gap, it is mitigated
                        if min_low <= top:
                            is_valid = False
                
                if is_valid:
                    fvgs.append({
                        'date': df.index[i+1], # Date of the big move
                        'top': top,
                        'bottom': bottom,
                        'avg': avg_price
                    })
        return fvgs

    @staticmethod
    def check_proximity(current_price, fvgs, threshold_pct):
        """Checks if current price is within X% of any FVG Top."""
        matches = []
        for fvg in fvgs:
            # Distance from FVG Top
            dist_pct = abs(current_price - fvg['top']) / current_price
            if dist_pct <= threshold_pct:
                matches.append(fvg)
        return matches

# --- HELPER FUNCTIONS ---

def get_gemini_response(prompt, api_key):
    """Interacts with Gemini 1.5 Flash."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini: {e}"

def fetch_financial_data(ticker_symbol):
    """Fetches base financial data."""
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    
    # Financial Statements (Wrapped in try/except for robustness)
    try:
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
    except:
        balance_sheet, income_stmt, cash_flow = None, None, None
    
    # Standard History for Charting (1 Year)
    history_daily = stock.history(period="1y", interval="1d")
    
    return stock, info, balance_sheet, income_stmt, cash_flow, history_daily

def get_multi_tf_data(ticker_symbol):
    """Fetches and resamples data for multi-timeframe FVG analysis."""
    stock = yf.Ticker(ticker_symbol)
    
    # 1. Fetch Base Data (Maximizing limits allowed by yfinance)
    df_1h = stock.history(period="730d", interval="1h") # Max available for hourly
    df_1d = stock.history(period="5y", interval="1d")   # Need long history for Year/Month resampling
    df_1mo = stock.history(period="max", interval="1mo") # Max history for monthly

    data_map = {}
    
    # 2. Process Timeframes
    if not df_1h.empty:
        data_map['1H'] = df_1h
        data_map['4H'] = FVG_Engine.resample_data(df_1h, '4h')
    
    if not df_1d.empty:
        data_map['1D'] = df_1d
        data_map['1W'] = FVG_Engine.resample_data(df_1d, 'W-FRI') # Weekly ending Friday
        
    if not df_1mo.empty:
        data_map['1M'] = df_1mo
        data_map['3M'] = FVG_Engine.resample_data(df_1mo, '3ME')  # Quarterly end
        data_map['6M'] = FVG_Engine.resample_data(df_1mo, '6ME')  # Semi-Annual end
        data_map['12M'] = FVG_Engine.resample_data(df_1mo, '12ME') # Yearly end

    return data_map

def get_social_buzz(query_term):
    """Scrapes news and text using DuckDuckGo."""
    results = []
    try:
        with DDGS() as ddgs:
            # News Search
            news_gen = ddgs.news(query_term, timelimit="w", max_results=3)
            for r in news_gen: results.append(f"Title: {r['title']} | Source: {r['source']}")
            
            # Text Search (Reddit/X context)
            text_gen = ddgs.text(f"{query_term} stock sentiment site:reddit.com OR site:x.com", timelimit="w", max_results=3)
            for r in text_gen: results.append(f"Snippet: {r['body']}")
    except Exception as e:
        results.append(f"Could not fetch live social data: {e}")
    return "\n".join(results)

# --- MAIN APPLICATION LOGIC ---

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Enter Company Ticker", placeholder="Type ticker here (e.g. MSFT)...")
with col2:
    run_btn = st.button("Initialize Analysis", use_container_width=True)

if run_btn and ticker_input:
    if not api_key:
        st.error("⚠️ API Key Missing. Please set it in sidebar or secrets.toml.")
        st.stop()

    full_ticker = f"{ticker_input.upper().strip()}{suffix}" if suffix else ticker_input.upper().strip()
    
    with st.spinner(f"Acquiring Target Data & Scanning Timeframes: {full_ticker}..."):
        # 1. Fetch Fundamentals
        stock, info, bs, inc, cf, history = fetch_financial_data(full_ticker)
        
        if history.empty:
            st.error(f"Could not fetch data for {full_ticker}. Check ticker/region.")
            st.stop()
            
        current_price = history['Close'].iloc[-1]
        
        # 2. Fetch & Process Multi-Timeframe Data for FVG
        tf_data = get_multi_tf_data(full_ticker)
        
        # 3. Define FVG Scan Criteria (Pairs & Proximity)
        # Format: (Pair Name, TF1, TF2, Prox1, Prox2)
        scan_pairs = [
            ("1H & 4H", "1H", "4H", 0.01, 0.01),
            ("1D & 1W", "1D", "1W", 0.02, 0.02),
            ("1W & 1M", "1W", "1M", 0.02, 0.03),
            ("1M & 3M", "1M", "3M", 0.03, 0.04),
            ("3M & 6M", "3M", "6M", 0.04, 0.05),
            ("6M & 12M", "6M", "12M", 0.05, 0.05),
        ]
        
        # 4. Run FVG Logic
        fvg_results = []
        for pair_name, tf1_name, tf2_name, prox1, prox2 in scan_pairs:
            # Verify data exists for both TFs
            if tf1_name in tf_data and tf2_name in tf_data and tf_data[tf1_name] is not None and tf_data[tf2_name] is not None:
                # Find FVGs
                fvgs_1 = FVG_Engine.find_bullish_fvgs(tf_data[tf1_name])
                fvgs_2 = FVG_Engine.find_bullish_fvgs(tf_data[tf2_name])
                
                # Check Proximity
                valid_1 = FVG_Engine.check_proximity(current_price, fvgs_1, prox1)
                valid_2 = FVG_Engine.check_proximity(current_price, fvgs_2, prox2)
                
                if valid_1 and valid_2:
                    fvg_results.append({
                        "pair": pair_name,
                        "status": "CONFLUENCE DETECTED",
                        "details": f"Price is within limits of Unmitigated Bullish FVGs on both {tf1_name} ({prox1*100}%) and {tf2_name} ({prox2*100}%).",
                        "zones": (valid_1, valid_2)
                    })
        
        # 5. Prepare Gemini Prompts
        
        # Fundamental Prompt
        fund_prompt = f"""
        You are an elite financial analyst. Analyze {full_ticker} (Price: {current_price}) based on these metrics:
        - Market Cap: {info.get('marketCap', 'N/A')}
        - P/E: {info.get('trailingPE', 'N/A')}, Forward P/E: {info.get('forwardPE', 'N/A')}
        - P/B: {info.get('priceToBook', 'N/A')}
        - PEG: {info.get('pegRatio', 'N/A')}
        - Div Yield: {info.get('dividendYield', 0)*100:.2f}%
        - Payout Ratio: {info.get('payoutRatio', 0)*100:.2f}%
        - Debt/Equity: {info.get('debtToEquity', 'N/A')}
        - Current Ratio: {info.get('currentRatio', 'N/A')}
        
        Evaluate strictly against these 3 methodologies:
        1. PETER LYNCH SCANNER (Fast Grower vs Stalwart vs Slow Grower)
        2. BENJAMIN GRAHAM SCANNER (Defensive vs Enterprising/Net-Net)
        3. MALKIEL & BERNSTEIN (Smart Beta: Value, Momentum, Low Vol, GARP)
        
        List out the metric evaluation for each. Conclude with a clear categorization.
        Use Markdown. Professional tone.
        """
        
        # Technical Prompt
        # Passing last 30 days of OHLCV for context
        tech_prompt = f"""
        You are a Chartered Market Technician (CMT). Analyze the technical structure of {full_ticker}.
        Current Price: {current_price}
        
        Recent OHLCV Data (Daily):
        {history.tail(30).to_csv()}

        Apply these SPECIFIC Logics:
        1. John Murphy: Trend definition (Above/Below SMAs?). Volume divergence?
        2. Richard Wyckoff: Signs of Accumulation (Springs, Tests) or Distribution (Upthrusts)?
        3. Steve Nison (Candlesticks): Hammers/Dojis at support? Shadow-to-body ratios?
        4. Al Brooks: Price Action. H2/L2 setups? Breakout failures/traps?
        5. Edwards & Magee: Geometric patterns? Head & Shoulders volume profile?

        Provide a "Sniper's Verdict": Bullish, Bearish, or Neutral/Wait.
        """
        
        # Social Prompt
        social_raw = get_social_buzz(f"${ticker_input} stock")
        social_prompt = f"""
        Analyze the sentiment for {full_ticker} based on these recent search snippets:
        {social_raw}
        
        Summarize the "Social Buzz":
        - Predominant Sentiment: Bullish/Bearish/Neutral
        - Key narrative driving the chatter.
        """

    # --- DISPLAY DASHBOARD ---
    st.subheader(f"{info.get('longName', full_ticker)} ({full_ticker})")
    
    # Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"{current_price:.2f}")
    m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
    m3.metric("52W High", f"{info.get('fiftyTwoWeekHigh', 0):.2f}")
    m4.metric("Recommendation", f"{info.get('recommendationKey', 'N/A').upper().replace('_', ' ')}")
    
    st.markdown("---")

    # --- AI GENERATION ---
    with st.spinner("🤖 Simulating Analyst Roundtable... (Fundamentals, Technicals, Sentiment)"):
        # Parallel execution isn't native in basic Streamlit without async, 
        # so we run sequential for stability.
        fund_analysis = get_gemini_response(fund_prompt, api_key)
        tech_analysis = get_gemini_response(tech_prompt, api_key)
        social_analysis = get_gemini_response(social_prompt, api_key)

    # --- TABS LAYOUT ---
    tab_fvg, tab_fund, tab_tech, tab_soc = st.tabs(["🎯 FVG Sniper", "🏛️ Fundamentals", "🔭 Technicals", "💬 Social"])
    
    with tab_fvg:
        st.markdown("### 🎯 Fair Value Gap (FVG) Confluence Scanner")
        st.markdown("""
        > **Strict Criteria:** Bullish FVGs Only | Unmitigated (Virgin) Zones Only | Dual Timeframe Confluence
        """)
        
        if fvg_results:
            for res in fvg_results:
                with st.expander(f"✅ {res['pair']} - CONFLUENCE DETECTED", expanded=True):
                    st.markdown(f"**Status:** <span class='status-pass'>{res['status']}</span>", unsafe_allow_html=True)
                    st.write(res['details'])
                    c_z1, c_z2 = st.columns(2)
                    with c_z1:
                        st.markdown(f"**{res['pair'].split('&')[0].strip()} Zone(s):**")
                        st.json(res['zones'][0])
                    with c_z2:
                        st.markdown(f"**{res['pair'].split('&')[1].strip()} Zone(s):**")
                        st.json(res['zones'][1])
        else:
            st.markdown("""
                <div style='padding: 20px; background-color: #2b2121; border-radius: 5px; border: 1px solid #da3633;'>
                    <h4 style='color: #da3633; margin:0;'>❌ No High-Probability Confluence Found</h4>
                    <p style='margin-top: 10px;'>The scan was run across all timeframe pairs (1H/4H, 1D/1W, etc.).<br>
                    No pairs currently show simultaneous <b>Unmitigated Bullish FVGs</b> within the strict proximity thresholds.</p>
                </div>
            """, unsafe_allow_html=True)

    with tab_fund:
        st.markdown("### 🏛️ Fundamental Valuation")
        st.markdown(fund_analysis)
        
    with tab_tech:
        st.markdown("### 🔭 Technical Structure")
        # Interactive Chart
        fig = go.Figure(data=[go.Candlestick(x=history.index, 
                                             open=history['Open'], 
                                             high=history['High'], 
                                             low=history['Low'], 
                                             close=history['Close'],
                                             name=full_ticker)])
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, title=f"{full_ticker} Daily Chart")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### AI Technical Commentary")
        st.markdown(tech_analysis)

    with tab_soc:
        st.markdown("### 🗣️ Market Sentiment")
        st.info(social_analysis)
        st.markdown("#### Raw Buzz Sources")
        with st.expander("View Source Snippets"):
            st.code(social_raw)