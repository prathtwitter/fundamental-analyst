import streamlit as st
import yfinance as yf
import google.generativeai as genai
from duckduckgo_search import DDGS
import pandas as pd
import plotly.graph_objects as go
import re
import warnings
import time

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Company Analyzer | Elite Intelligence",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress deprecation warnings
warnings.filterwarnings("ignore")

# --- LUXURY STYLING (CSS) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #d4af37 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
    .stMetricValue { color: #d4af37 !important; }
    .stButton>button { background-color: #1f6feb; color: white; border: none; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #388bfd; }
    .status-pass { color: #2ea043; font-weight: bold; }
    .status-fail { color: #da3633; font-weight: bold; }
    /* Improve input visibility */
    .stTextInput>div>div>input { color: #ffffff; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("🔒 API Key Loaded Securely")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    region_selection = st.selectbox("Select Market Region", ["USA 🇺🇸", "India 🇮🇳", "Canada 🇨🇦"])
    region_map = {"USA 🇺🇸": "USA", "India 🇮🇳": "India", "Canada 🇨🇦": "Canada"}
    selected_country = region_map[region_selection]
    
    st.info("💡 **Ticker Guide:**\n- USA: `TSLA`, `AAPL`\n- India: `RELIANCE.NS`, `TCS.NS`\n- Canada: `SHOP.TO`, `RY.TO`")

# --- MAIN HEADER ---
st.title("♟️ Company Analyzer")
st.markdown(f"*Strategic Deep Dive | Region: {selected_country}*")
st.markdown("---")

# --- SESSION STATE ---
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = None

# --- FVG ENGINE (STRICT MATCH) ---
class FVG_Engine:
    @staticmethod
    def resample_data(df, interval):
        if df is None or df.empty: return None
        logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        try:
            return df.resample(interval).agg(logic).dropna()
        except Exception:
            return None

    @staticmethod
    def find_bullish_fvgs(df, strict_mitigation=True):
        fvgs = []
        if df is None or len(df) < 3: return fvgs

        for i in range(len(df) - 2):
            c1 = df.iloc[i]
            c2 = df.iloc[i+1] # Displacement
            c3 = df.iloc[i+2]
            
            # Gap exists: C3 Low > C1 High
            if c3['Low'] > c1['High'] and c2['Close'] > c2['Open']:
                top = c3['Low']
                bottom = c1['High']
                avg_price = (top + bottom) / 2
                is_valid = True
                
                if strict_mitigation:
                    future_candles = df.iloc[i+3:]
                    if not future_candles.empty:
                        min_low = future_candles['Low'].min()
                        if min_low <= top: is_valid = False # Mitigated
                
                if is_valid:
                    fvgs.append({'date': df.index[i+1], 'top': top, 'bottom': bottom, 'avg': avg_price})
        return fvgs

    @staticmethod
    def check_proximity(current_price, fvgs, threshold_pct):
        matches = []
        for fvg in fvgs:
            dist_pct = abs(current_price - fvg['top']) / current_price
            if dist_pct <= threshold_pct:
                matches.append(fvg)
        return matches

# --- HELPER FUNCTIONS ---
def search_tickers_safe(query, country):
    """Searches using DuckDuckGo with Ratelimit Handling."""
    candidates = []
    search_term = f"site:finance.yahoo.com/quote {query} {country} stock"
    try:
        # Sleep briefly to avoid hammering the API if user clicks fast
        time.sleep(1) 
        with DDGS() as ddgs:
            results = list(ddgs.text(search_term, max_results=4))
            for r in results:
                url = r['href']
                title = r['title']
                match = re.search(r'finance\.yahoo\.com/quote/([A-Z0-9.-]+)', url)
                if match:
                    ticker = match.group(1)
                    if not any(d['symbol'] == ticker for d in candidates):
                        candidates.append({'symbol': ticker, 'title': title})
    except Exception as e:
        st.error(f"Search Engine Busy (Rate Limit). Switching to manual input.")
        return []
    return candidates

def get_gemini_response(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"**Error:** {e}"

def fetch_financial_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    try:
        info = stock.info
    except:
        info = {}
    history_daily = stock.history(period="1y", interval="1d")
    if history_daily.empty:
        return None, None, None, None, None, None
    return stock, info, None, None, None, history_daily

def get_multi_tf_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    df_1h = stock.history(period="730d", interval="1h") 
    df_1d = stock.history(period="5y", interval="1d")   
    df_1mo = stock.history(period="max", interval="1mo") 

    data_map = {}
    if not df_1h.empty:
        data_map['1H'] = df_1h
        data_map['4H'] = FVG_Engine.resample_data(df_1h, '4h')
    if not df_1d.empty:
        data_map['1D'] = df_1d
        data_map['1W'] = FVG_Engine.resample_data(df_1d, 'W-FRI') 
    if not df_1mo.empty:
        data_map['1M'] = df_1mo
        data_map['3M'] = FVG_Engine.resample_data(df_1mo, '3ME')  
        data_map['6M'] = FVG_Engine.resample_data(df_1mo, '6ME')  
        data_map['12M'] = FVG_Engine.resample_data(df_1mo, '12ME') 
    return data_map

def get_social_buzz(query_term):
    try:
        with DDGS() as ddgs:
            news = list(ddgs.news(query_term, timelimit="w", max_results=3))
            return "\n".join([f"News: {n['title']}" for n in news])
    except:
        return "Social buzz unavailable (Rate Limit)."

# --- REDESIGNED WORKFLOW ---

st.markdown("### 1️⃣ Identify Target")

col_mode, col_input = st.columns([1, 3])

with col_mode:
    # Toggle between Search and Direct Input
    input_mode = st.radio("Input Method:", ["Search by Name", "Direct Ticker Input"])

ticker_to_analyze = None

with col_input:
    if input_mode == "Search by Name":
        c_search_1, c_search_2 = st.columns([3, 1])
        with c_search_1:
            search_query = st.text_input("Company Name", placeholder="e.g. Tesla, Reliance...")
        with c_search_2:
            st.write("") # Spacer
            st.write("") 
            if st.button("🔎 Find", use_container_width=True):
                if search_query:
                    with st.spinner("Scanning..."):
                        results = search_tickers_safe(search_query, selected_country)
                        st.session_state.search_results = results
        
        # Display Search Results
        if st.session_state.search_results:
            options = [f"{item['symbol']} | {item['title']}" for item in st.session_state.search_results]
            selected_option = st.selectbox("Select Correct Stock:", options)
            if selected_option:
                ticker_to_analyze = selected_option.split(" | ")[0]
        elif search_query and not st.session_state.search_results:
             st.warning("Search returned no results or was blocked. Please switch to 'Direct Ticker Input'.")
             
    else: # Direct Input Mode
        direct_input = st.text_input("Enter Exact Ticker Symbol", placeholder="e.g. TSLA, RELIANCE.NS, SHOP.TO")
        if direct_input:
            ticker_to_analyze = direct_input.strip().upper()

# --- ANALYSIS EXECUTION ---
if ticker_to_analyze:
    st.divider()
    st.markdown(f"### 2️⃣ Analyze: **{ticker_to_analyze}**")
    
    if st.button("🚀 Launch Strategic Deep Dive", use_container_width=True):
        if not api_key:
            st.error("⚠️ API Key Missing.")
            st.stop()
            
        with st.spinner(f"Running Multi-Timeframe FVG Scan & AI Analysis on {ticker_to_analyze}..."):
            
            # 1. Fetch Data
            stock, info, bs, inc, cf, history = fetch_financial_data(ticker_to_analyze)
            if history is None:
                st.error(f"❌ Could not load data for {ticker_to_analyze}. Check if ticker is correct for Region: {selected_country}.")
                st.stop()
            
            current_price = history['Close'].iloc[-1]
            
            # 2. FVG Scan
            tf_data = get_multi_tf_data(ticker_to_analyze)
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
                if tf1_name in tf_data and tf2_name in tf_data and tf_data[tf1_name] is not None:
                    fvgs_1 = FVG_Engine.find_bullish_fvgs(tf_data[tf1_name])
                    fvgs_2 = FVG_Engine.find_bullish_fvgs(tf_data[tf2_name])
                    valid_1 = FVG_Engine.check_proximity(current_price, fvgs_1, prox1)
                    valid_2 = FVG_Engine.check_proximity(current_price, fvgs_2, prox2)
                    
                    if valid_1 and valid_2:
                        fvg_results.append({
                            "pair": pair_name,
                            "status": "CONFLUENCE DETECTED",
                            "details": f"Price within limits ({prox1*100}% & {prox2*100}%) of Unmitigated Bullish FVGs.",
                            "zones": (valid_1, valid_2)
                        })

            # 3. AI Analysis
            fund_prompt = f"""
            Analyze {ticker_to_analyze} (Price: {current_price}) Metrics:
            - Market Cap: {info.get('marketCap', 'N/A')}
            - P/E: {info.get('trailingPE', 'N/A')}
            - PEG: {info.get('pegRatio', 'N/A')}
            - P/B: {info.get('priceToBook', 'N/A')}
            
            Apply: 1. Peter Lynch 2. Ben Graham 3. Malkiel/Bernstein.
            Output: Concise Markdown.
            """
            tech_prompt = f"""
            Analyze Technicals {ticker_to_analyze} (Price: {current_price}).
            Last 30 Days OHLCV: {history.tail(30).to_csv()}
            Apply: Murphy, Wyckoff, Nison, Brooks.
            Verdict: Bullish/Bearish/Neutral?
            """
            
            soc_buzz = get_social_buzz(f"{ticker_to_analyze} stock")
            soc_prompt = f"Sentiment Analysis based on: {soc_buzz}"
            
            fund_an = get_gemini_response(fund_prompt, api_key)
            tech_an = get_gemini_response(tech_prompt, api_key)
            soc_an = get_gemini_response(soc_prompt, api_key)

            # --- DISPLAY RESULTS ---
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"{current_price:.2f} {info.get('currency', '')}")
            m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            m3.metric("Rec Key", f"{info.get('recommendationKey', 'N/A')}")
            
            t1, t2, t3, t4 = st.tabs(["🎯 FVG Sniper", "🏛️ Fundamentals", "🔭 Technicals", "💬 Social"])
            
            with t1:
                st.markdown("### 🎯 FVG Confluence")
                st.info("Filters: Bullish Only | Unmitigated (Strict) | Dual Timeframe Proximity")
                if fvg_results:
                    for res in fvg_results:
                        with st.expander(f"✅ {res['pair']} - CONFLUENCE FOUND", expanded=True):
                            st.write(res['details'])
                            c_z1, c_z2 = st.columns(2)
                            with c_z1: st.json(res['zones'][0])
                            with c_z2: st.json(res['zones'][1])
                else:
                    st.error("❌ No Confluence Found matching Strict Criteria.")
            
            with t2: st.markdown(fund_an)
            with t3:
                fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'])])
                fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(tech_an)
            with t4: st.markdown(soc_an)