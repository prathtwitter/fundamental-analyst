import streamlit as st
import yfinance as yf
import google.generativeai as genai
from duckduckgo_search import DDGS
import pandas as pd
import plotly.graph_objects as go
import re
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
    
    # 1. API Key Logic
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("🔒 API Key Loaded Securely")
    else:
        api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key")
    
    st.markdown("---")
    region_selection = st.selectbox("Select Market Region", ["USA 🇺🇸", "India 🇮🇳", "Canada 🇨🇦"])
    
    # Region mapping for search context
    region_map = {"USA 🇺🇸": "USA", "India 🇮🇳": "India", "Canada 🇨🇦": "Canada"}
    selected_country = region_map[region_selection]

# --- MAIN HEADER ---
st.title("♟️ Company Analyzer")
st.markdown(f"*Strategic Deep Dive | Region: {selected_country}*")
st.markdown("---")

# --- SESSION STATE INITIALIZATION ---
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

# --- FVG ENGINE (Strategic Sniper Logic) ---
class FVG_Engine:
    @staticmethod
    def resample_data(df, interval):
        if df is None or df.empty: return None
        # Drop NaN to ensure clean candles
        logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        try:
            resampled = df.resample(interval).agg(logic).dropna()
            return resampled
        except Exception as e:
            return None

    @staticmethod
    def find_bullish_fvgs(df, strict_mitigation=True):
        """
        Identifies Bullish FVGs (Candle 1 High < Candle 3 Low).
        STRICT CRITERIA:
        - Current forming candle is ignored.
        - Invalid if ANY future candle wicks into the gap (Low <= Top).
        """
        fvgs = []
        if df is None or len(df) < 3: return fvgs

        # Iterate up to len-1 to ignore the very last forming candle if needed
        # We stop at len - 2 because we need i, i+1, i+2 closed candles
        for i in range(len(df) - 2):
            c1 = df.iloc[i]
            c2 = df.iloc[i+1] # The displacement candle
            c3 = df.iloc[i+2]
            
            # 1. Basic Structure: Gap Exists AND Displacement Candle is Green
            if c3['Low'] > c1['High'] and c2['Close'] > c2['Open']:
                top = c3['Low']
                bottom = c1['High']
                avg_price = (top + bottom) / 2
                
                is_valid = True
                
                # 2. Strict Mitigation Check
                if strict_mitigation:
                    # Check ALL subsequent candles after C3
                    future_candles = df.iloc[i+3:]
                    if not future_candles.empty:
                        min_low = future_candles['Low'].min()
                        # If any future Low touches or goes below the Top of the FVG, it is mitigated
                        if min_low <= top:
                            is_valid = False
                
                if is_valid:
                    fvgs.append({'date': df.index[i+1], 'top': top, 'bottom': bottom, 'avg': avg_price})
        return fvgs

    @staticmethod
    def check_proximity(current_price, fvgs, threshold_pct):
        matches = []
        for fvg in fvgs:
            # Distance from FVG Top (Support Level)
            dist_pct = abs(current_price - fvg['top']) / current_price
            if dist_pct <= threshold_pct:
                matches.append(fvg)
        return matches

# --- HELPER FUNCTIONS ---

def search_tickers(query, country):
    """
    Searches DuckDuckGo for Yahoo Finance URLs and extracts tickers.
    Returns a list of candidate dictionaries.
    """
    candidates = []
    # Broad search query to get Yahoo Finance quote pages
    search_term = f"site:finance.yahoo.com/quote {query} {country} stock"
    
    try:
        with DDGS() as ddgs:
            # Get more results to ensure we catch the right one
            results = list(ddgs.text(search_term, max_results=5))
            
            for r in results:
                url = r['href']
                title = r['title']
                
                # Extract Ticker from URL: finance.yahoo.com/quote/TICKER?p=...
                # Regex looks for the segment after /quote/ and before / or ?
                match = re.search(r'finance\.yahoo\.com/quote/([A-Z0-9.-]+)', url)
                
                if match:
                    ticker = match.group(1)
                    # Avoid duplicates
                    if not any(d['symbol'] == ticker for d in candidates):
                        candidates.append({'symbol': ticker, 'title': title})
                        
    except Exception as e:
        return []
    
    return candidates

def get_gemini_response(prompt, api_key):
    """Robust AI Call function."""
    try:
        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"**Error connecting to Gemini:** {e}"

def fetch_financial_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    try:
        info = stock.info
        # Basic check if ticker is valid
        if 'symbol' not in info and 'longName' not in info:
             # Fallback: sometimes info is empty but history works.
             # We rely on history check later.
             pass
    except:
        info = {}

    try:
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
    except:
        balance_sheet, income_stmt, cash_flow = None, None, None
    
    # Get 1 year daily history for chart
    try:
        history_daily = stock.history(period="1y", interval="1d")
    except:
        history_daily = pd.DataFrame()
        
    return stock, info, balance_sheet, income_stmt, cash_flow, history_daily

def get_multi_tf_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    
    # Fetch base datasets
    df_1h = stock.history(period="730d", interval="1h") 
    df_1d = stock.history(period="5y", interval="1d")   
    df_1mo = stock.history(period="max", interval="1mo") 

    data_map = {}
    
    # Process Hourly & 4H
    if not df_1h.empty:
        data_map['1H'] = df_1h
        data_map['4H'] = FVG_Engine.resample_data(df_1h, '4h')
    
    # Process Daily & Weekly
    if not df_1d.empty:
        data_map['1D'] = df_1d
        data_map['1W'] = FVG_Engine.resample_data(df_1d, 'W-FRI') 
    
    # Process Monthly & larger
    if not df_1mo.empty:
        data_map['1M'] = df_1mo
        data_map['3M'] = FVG_Engine.resample_data(df_1mo, '3ME')  
        data_map['6M'] = FVG_Engine.resample_data(df_1mo, '6ME')  
        data_map['12M'] = FVG_Engine.resample_data(df_1mo, '12ME') 
        
    return data_map

def get_social_buzz(query_term):
    results = []
    try:
        with DDGS() as ddgs:
            news_gen = ddgs.news(query_term, timelimit="w", max_results=3)
            for r in news_gen: results.append(f"Title: {r['title']} | Source: {r['source']}")
            text_gen = ddgs.text(f"{query_term} stock sentiment site:reddit.com OR site:x.com", timelimit="w", max_results=3)
            for r in text_gen: results.append(f"Snippet: {r['body']}")
    except Exception as e:
        results.append(f"Could not fetch live social data: {e}")
    return "\n".join(results)

# --- SEARCH & SELECTION WORKFLOW ---

st.markdown("### 1️⃣ Find Company")
c1, c2 = st.columns([3, 1])
with c1:
    search_input = st.text_input("Enter Company Name or Ticker", placeholder="e.g. Microsoft, Reliance, MSFT...", key="search_input")
with c2:
    if st.button("🔎 Search Symbol", use_container_width=True):
        if search_input:
            with st.spinner(f"Scanning for '{search_input}' in {selected_country}..."):
                results = search_tickers(search_input, selected_country)
                st.session_state.search_results = results
                # Reset previous analysis if new search
                st.session_state.selected_ticker = None 
        else:
            st.warning("Please enter a name to search.")

# Display Results if Available
ticker_to_analyze = None

if st.session_state.search_results:
    st.markdown("### 2️⃣ Select Correct Stock")
    
    # Create a friendly list of strings "Symbol - Title"
    options = [f"{item['symbol']} | {item['title']}" for item in st.session_state.search_results]
    
    selected_option = st.selectbox("Choose the correct match:", options)
    
    # Extract the symbol back from the selection string
    if selected_option:
        ticker_to_analyze = selected_option.split(" | ")[0]
        st.info(f"Selected: **{ticker_to_analyze}**")

# --- EXECUTION BUTTON ---

run_analysis = False
if ticker_to_analyze:
    st.markdown("### 3️⃣ Execute Analysis")
    if st.button("🚀 Run Deep Dive Analysis", use_container_width=True):
        run_analysis = True

# --- ANALYSIS ENGINE ---

if run_analysis and ticker_to_analyze:
    if not api_key:
        st.error("⚠️ API Key Missing. Please set it in sidebar or secrets.toml.")
        st.stop()
    
    with st.spinner(f"Acquiring Intelligence on {ticker_to_analyze}..."):
        
        # 1. Fetch Fundamentals
        stock, info, bs, inc, cf, history = fetch_financial_data(ticker_to_analyze)
        
        if history.empty:
            st.error(f"Could not retrieve historical data for **{ticker_to_analyze}**. It may be delisted or inactive.")
            st.stop()
            
        current_price = history['Close'].iloc[-1]
        
        # 2. FVG Scan
        tf_data = get_multi_tf_data(ticker_to_analyze)
        
        # Pairs & Tolerances
        scan_pairs = [
            ("1H & 4H", "1H", "4H", 0.01, 0.01),
            ("1D & 1W", "1D", "1W", 0.02, 0.02),
            ("1W & 1M", "1W", "1M", 0.02, 0.03),
            ("1M & 3M", "1M", "3M", 0.03, 0.04),
            ("3M & 6M", "3M", "6M", 0.04, 0.05),
            ("6M & 12M", "6M", "12M", 0.05, 0.05),
        ]
        
        fvg_results = []
        for pair_name, tf1_name, tf2_name, prox1, prox2 in scan_pairs:
            if tf1_name in tf_data and tf2_name in tf_data and tf_data[tf1_name] is not None and tf_data[tf2_name] is not None:
                fvgs_1 = FVG_Engine.find_bullish_fvgs(tf_data[tf1_name])
                fvgs_2 = FVG_Engine.find_bullish_fvgs(tf_data[tf2_name])
                
                valid_1 = FVG_Engine.check_proximity(current_price, fvgs_1, prox1)
                valid_2 = FVG_Engine.check_proximity(current_price, fvgs_2, prox2)
                
                if valid_1 and valid_2:
                    fvg_results.append({
                        "pair": pair_name,
                        "status": "CONFLUENCE DETECTED",
                        "details": f"Price is within limits ({prox1*100}% & {prox2*100}%) of Unmitigated Bullish FVGs.",
                        "zones": (valid_1, valid_2)
                    })
        
        # 3. Prepare Prompts
        fund_prompt = f"""
        You are an elite financial analyst. Analyze {ticker_to_analyze} (Price: {current_price}) based on these metrics:
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
        
        tech_prompt = f"""
        You are a Chartered Market Technician (CMT). Analyze the technical structure of {ticker_to_analyze}.
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
        
        social_raw = get_social_buzz(f"{ticker_to_analyze} stock")
        social_prompt = f"""
        Analyze the sentiment for {ticker_to_analyze} based on these recent search snippets:
        {social_raw}
        
        Summarize the "Social Buzz":
        - Predominant Sentiment: Bullish/Bearish/Neutral
        - Key narrative driving the chatter.
        """

    # --- DISPLAY DASHBOARD ---
    st.divider()
    st.subheader(f"📊 Report: {info.get('longName', ticker_to_analyze)} ({ticker_to_analyze})")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"{current_price:.2f} {info.get('currency', '')}")
    m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
    m3.metric("52W High", f"{info.get('fiftyTwoWeekHigh', 0):.2f}")
    m4.metric("Recommendation", f"{info.get('recommendationKey', 'N/A').upper().replace('_', ' ')}")
    
    st.markdown("---")

    with st.spinner("🤖 Simulating Analyst Roundtable..."):
        fund_analysis = get_gemini_response(fund_prompt, api_key)
        tech_analysis = get_gemini_response(tech_prompt, api_key)
        social_analysis = get_gemini_response(social_prompt, api_key)

    tab_fvg, tab_fund, tab_tech, tab_soc = st.tabs(["🎯 FVG Sniper", "🏛️ Fundamentals", "🔭 Technicals", "💬 Social"])
    
    with tab_fvg:
        st.markdown("### 🎯 Fair Value Gap (FVG) Confluence Scanner")
        st.markdown("> **Strict Criteria:** Bullish FVGs Only | Unmitigated (Virgin) Zones Only | Dual Timeframe Confluence")
        
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
        fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name=ticker_to_analyze)])
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, title=f"{ticker_to_analyze} Daily Chart")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### AI Technical Commentary")
        st.markdown(tech_analysis)

    with tab_soc:
        st.markdown("### 🗣️ Market Sentiment")
        st.info(social_analysis)
        st.markdown("#### Raw Buzz Sources")
        with st.expander("View Source Snippets"):
            st.code(social_raw)