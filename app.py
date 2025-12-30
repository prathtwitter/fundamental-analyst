import streamlit as st
import yfinance as yf
import google.generativeai as genai
from duckduckgo_search import DDGS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import time
import json
import difflib

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
    .stTextInput>div>div>input { color: #ffffff; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # API Key Handling
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("🔒 API Key Loaded Securely")
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("🔒 API Key Loaded Securely")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.info("💡 **Global Search Enabled**\nSystem searches US, India, and Canada databases simultaneously.")

# --- GEMINI API SETUP ---
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
    except Exception as e:
        st.sidebar.error(f"Model Error: {e}")

# --- DATA LOADING & SEARCH ENGINE ---
@st.cache_data
def load_and_prep_data():
    """
    Loads US, India, and Canada stock lists from CSVs.
    Standardizes columns and adds Yahoo Finance suffixes.
    """
    master_df = pd.DataFrame()

    try:
        # Load India (Add .NS)
        india = pd.read_csv("india_stocks.csv")
        india.columns = [c.strip() for c in india.columns] # Clean column names
        india['Country'] = 'India'
        # Ensure 'Symbol' exists, fallback to first column if not
        sym_col = 'Symbol' if 'Symbol' in india.columns else india.columns[0]
        name_col = 'Company Name' if 'Company Name' in india.columns else india.columns[1]
        india['Yahoo_Ticker'] = india[sym_col].astype(str).str.strip() + ".NS"
        india['Display_Name'] = india[name_col].astype(str).str.strip()
        master_df = pd.concat([master_df, india[['Yahoo_Ticker', 'Display_Name', 'Country']]])
    except Exception:
        pass # Fail silently if file missing

    try:
        # Load Canada (Add .TO)
        canada = pd.read_csv("canada_stocks.csv")
        canada.columns = [c.strip() for c in canada.columns]
        canada['Country'] = 'Canada'
        sym_col = 'Ticker' if 'Ticker' in canada.columns else 'Symbol'
        name_col = 'Company Name' if 'Company Name' in canada.columns else canada.columns[1]
        # Clean Tickers (remove .TO if exists to avoid double, then add back)
        canada['Yahoo_Ticker'] = canada[sym_col].astype(str).str.replace('.TO', '', regex=False).str.strip() + ".TO"
        canada['Display_Name'] = canada[name_col].astype(str).str.strip()
        master_df = pd.concat([master_df, canada[['Yahoo_Ticker', 'Display_Name', 'Country']]])
    except Exception:
        pass

    try:
        # Load USA (No Suffix)
        usa = pd.read_csv("us_stocks.csv")
        usa.columns = [c.strip() for c in usa.columns]
        usa['Country'] = 'USA'
        sym_col = 'Ticker' if 'Ticker' in usa.columns else 'Symbol'
        name_col = 'Company Name' if 'Company Name' in usa.columns else usa.columns[1]
        usa['Yahoo_Ticker'] = usa[sym_col].astype(str).str.strip()
        usa['Display_Name'] = usa[name_col].astype(str).str.strip()
        master_df = pd.concat([master_df, usa[['Yahoo_Ticker', 'Display_Name', 'Country']]])
    except Exception:
        pass
    
    return master_df

def smart_search(query, df):
    """
    Performs fuzzy search on Ticker and Company Name.
    Returns top matches.
    """
    if df.empty or not query:
        return []
    
    query = query.lower().strip()
    
    # 1. Exact/Substring Match (Fastest)
    mask = (
        df['Yahoo_Ticker'].str.lower().str.contains(query, na=False) | 
        df['Display_Name'].str.lower().str.contains(query, na=False)
    )
    direct_matches = df[mask].head(15).to_dict('records')
    
    # 2. Fuzzy Match (If few direct matches)
    if len(direct_matches) < 5:
        # Create a list of names for difflib
        all_names = df['Display_Name'].tolist()
        # Find close matches to the query
        close_names = difflib.get_close_matches(query, all_names, n=5, cutoff=0.5)
        
        # Retrieve rows for these names
        fuzzy_rows = df[df['Display_Name'].isin(close_names)].to_dict('records')
        
        # Combine and deduplicate
        seen = set(d['Yahoo_Ticker'] for d in direct_matches)
        for item in fuzzy_rows:
            if item['Yahoo_Ticker'] not in seen:
                direct_matches.append(item)
                
    return direct_matches

# Load data once
stock_db = load_and_prep_data()

# --- MAIN HEADER ---
st.title("♟️ Company Analyzer")
st.markdown("*Strategic Deep Dive | Global Markets*")
st.markdown("---")

# --- SESSION STATE ---
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = None


# ==========================================
# FVG ENGINE - EXACT COPY FROM WORKING FVG SCANNER
# ==========================================

def calculate_bullish_fvgs(df):
    """
    Identifies ONLY Unmitigated Bullish FVGs (Support).
    """
    if df is None or len(df) < 4: 
        return []

    highs = df['High'].values
    lows = df['Low'].values
    n = len(lows)
    
    min_future_low = np.full(n, np.inf)
    current_min = np.inf
    
    for i in range(n-2, -1, -1):
        current_min = min(current_min, lows[i])
        min_future_low[i] = current_min

    unmitigated_fvgs = []
    scan_limit = n - 3
    if scan_limit < 0: return []

    for i in range(scan_limit):
        if highs[i] < lows[i+2]:
            bottom = highs[i]
            top = lows[i+2]
            
            if i + 3 <= n - 2:
                if min_future_low[i+3] <= top:
                    continue  # Mitigated
            
            unmitigated_fvgs.append((bottom, top))

    return unmitigated_fvgs


def calculate_bearish_fvgs(df):
    """
    Identifies ONLY Unmitigated Bearish FVGs (Resistance).
    """
    if df is None or len(df) < 4: 
        return []

    highs = df['High'].values
    lows = df['Low'].values
    n = len(highs)
    
    max_future_high = np.full(n, -np.inf)
    current_max = -np.inf
    
    for i in range(n-2, -1, -1):
        current_max = max(current_max, highs[i])
        max_future_high[i] = current_max

    unmitigated_fvgs = []
    scan_limit = n - 3
    if scan_limit < 0: return []

    for i in range(scan_limit):
        if lows[i] > highs[i+2]:
            top = lows[i]
            bottom = highs[i+2]
            
            if i + 3 <= n - 2:
                if max_future_high[i+3] >= bottom:
                    continue  # Mitigated
            
            unmitigated_fvgs.append((bottom, top))

    return unmitigated_fvgs


def is_near_fvg(price, fvgs, threshold_pct):
    if not fvgs: return False, None
    threshold = threshold_pct / 100.0
    
    for bot, top in fvgs:
        if bot <= price <= top:
            return True, {'bottom': bot, 'top': top, 'status': 'INSIDE'}
        
        dist_to_top = abs(price - top)
        dist_to_bot = abs(price - bot)
        min_dist = min(dist_to_top, dist_to_bot)
        
        if (min_dist / price) <= threshold:
            return True, {'bottom': bot, 'top': top, 'status': 'NEAR', 'distance_pct': (min_dist / price) * 100}
            
    return False, None


def resample_custom(df, timeframe):
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    
    if timeframe == '1D': return df.resample('1D').agg(agg_dict).dropna()
    if timeframe == '4H': return df.resample('4h').agg(agg_dict).dropna()
    if timeframe == '1W': return df.resample('W').agg(agg_dict).dropna()
    if timeframe == '1M': return df.resample('ME').agg(agg_dict).dropna()
        
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    
    if timeframe == '3M':
        df['Group'] = (df['Month'] - 1) // 3
        grouper = ['Year', 'Group']
    elif timeframe == '6M':
        df['Group'] = (df['Month'] - 1) // 6
        grouper = ['Year', 'Group']
    elif timeframe == '12M':
        df['Group'] = 0 
        grouper = ['Year']
    else:
        return df 

    resampled = df.groupby(grouper).agg(agg_dict).dropna()
    resampled = resampled.sort_index()
    return resampled


# ==========================================
# ANALYSIS ENGINE 
# ==========================================

def analyze_fvg_confluence(ticker, df_1h, df_1d, df_1mo):
    if df_1d is None or df_1d.empty: return None
    current_price = df_1d['Close'].iloc[-1]
    
    data_frames = {}
    if df_1h is not None and not df_1h.empty:
        data_frames['1H'] = df_1h
        data_frames['4H'] = resample_custom(df_1h, '4H')
    data_frames['1D'] = df_1d
    data_frames['1W'] = resample_custom(df_1d, '1W')
    if df_1mo is not None and not df_1mo.empty:
        data_frames['1M'] = df_1mo
        data_frames['3M'] = resample_custom(df_1mo, '3M')
        data_frames['6M'] = resample_custom(df_1mo, '6M')
        data_frames['12M'] = resample_custom(df_1mo, '12M')
    
    bullish_fvgs = {}
    bearish_fvgs = {}
    for tf, df in data_frames.items():
        if df is not None and not df.empty:
            bullish_fvgs[tf] = calculate_bullish_fvgs(df)
            bearish_fvgs[tf] = calculate_bearish_fvgs(df)
    
    confluence_pairs = [
        ('1H', '4H', 1.0, 1.0), ('1D', '1W', 2.0, 2.0), ('1W', '1M', 2.0, 3.0),
        ('1M', '3M', 3.0, 4.0), ('3M', '6M', 4.0, 5.0), ('6M', '12M', 5.0, 5.0),
    ]
    
    results = {
        'current_price': current_price,
        'bullish_confluence': {}, 'bearish_confluence': {},
        'bullish_fvg_counts': {}, 'bearish_fvg_counts': {},
    }
    
    for tf in data_frames.keys():
        results['bullish_fvg_counts'][tf] = len(bullish_fvgs.get(tf, []))
        results['bearish_fvg_counts'][tf] = len(bearish_fvgs.get(tf, []))
    
    for tf1, tf2, thresh1, thresh2 in confluence_pairs:
        pair_name = f"{tf1}, {tf2}"
        if tf1 in bullish_fvgs and tf2 in bullish_fvgs:
            near1, zone1 = is_near_fvg(current_price, bullish_fvgs[tf1], thresh1)
            near2, zone2 = is_near_fvg(current_price, bullish_fvgs[tf2], thresh2)
            if near1 and near2:
                results['bullish_confluence'][pair_name] = {'detected': True, 'tf1_zone': zone1, 'tf2_zone': zone2, 'thresholds': (thresh1, thresh2)}
        
        if tf1 in bearish_fvgs and tf2 in bearish_fvgs:
            near1, zone1 = is_near_fvg(current_price, bearish_fvgs[tf1], thresh1)
            near2, zone2 = is_near_fvg(current_price, bearish_fvgs[tf2], thresh2)
            if near1 and near2:
                results['bearish_confluence'][pair_name] = {'detected': True, 'tf1_zone': zone1, 'tf2_zone': zone2, 'thresholds': (thresh1, thresh2)}
    
    return results


class MathWiz:
    @staticmethod
    def identify_strict_swings(df, neighbor_count=3):
        is_swing_high = pd.Series(True, index=df.index)
        is_swing_low = pd.Series(True, index=df.index)
        for i in range(1, neighbor_count + 1):
            is_swing_high &= (df['High'] > df['High'].shift(i))
            is_swing_low &= (df['Low'] < df['Low'].shift(i))
            is_swing_high &= (df['High'] > df['High'].shift(-i))
            is_swing_low &= (df['Low'] < df['Low'].shift(-i))
        return is_swing_high, is_swing_low
    
    @staticmethod
    def check_ifvg_reversal(df):
        subset = df.iloc[-5:].copy().dropna()
        if len(subset) < 5: return None
        c1, c3, c5 = subset.iloc[0], subset.iloc[2], subset.iloc[4]
        if c3['High'] < c1['Low'] and c5['Low'] > c3['High']: return 'Bull'
        if c3['Low'] > c1['High'] and c5['High'] < c3['Low']: return 'Bear'
        return None

    @staticmethod
    def check_consecutive_candles(df, num_candles):
        if len(df) < num_candles: return None
        recent = df.iloc[-num_candles:]
        if all(recent['Close'] < recent['Open']): return 'Bull'
        elif all(recent['Close'] > recent['Open']): return 'Bear'
        return None

    @staticmethod
    def calculate_choppiness(high, low, close, length=14):
        try:
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            range_diff = high.rolling(window=length).max() - low.rolling(window=length).min()
            range_diff.replace(0, np.nan, inplace=True)
            return 100 * (np.log10(tr.rolling(window=length).sum() / range_diff) / np.log10(length))
        except:
            return pd.Series(dtype='float64')


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_gemini_response(prompt):
    if model is None: return "**Error:** Model not initialized. Please provide API key."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"**Error:** {e}"

def fetch_financial_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    try: info = stock.info
    except: info = {}
    history_1h = stock.history(period="730d", interval="1h")
    history_daily = stock.history(period="max", interval="1d")
    history_monthly = stock.history(period="max", interval="1mo")
    if history_daily.empty: return None, None, None, None, None
    return stock, info, history_1h, history_daily, history_monthly

def get_social_sentiment_data(ticker, company_name=""):
    all_data = []
    try:
        time.sleep(0.5)
        with DDGS() as ddgs:
            try:
                news = list(ddgs.news(f"{ticker} stock {company_name}", timelimit="w", max_results=5))
                for n in news: all_data.append({"type": "News", "title": n.get('title'), "source": n.get('source'), "body": n.get('body', '')[:200]})
            except: pass
    except Exception: pass
    return all_data


# ==========================================
# UI WORKFLOW
# ==========================================
st.markdown("### 1️⃣ Identify Target")

# --- SMART SEARCH UI ---
col_search, col_btn = st.columns([3, 1])
ticker_to_analyze = None

with col_search:
    user_query = st.text_input("Search Company or Ticker", placeholder="e.g. Reliance, Apple, Tesla (Typos allowed)")

if user_query:
    if stock_db is not None and not stock_db.empty:
        results = smart_search(user_query, stock_db)
        
        if results:
            # Format options: "Ticker | Name [Country]"
            options = [f"{r['Yahoo_Ticker']} | {r['Display_Name']} [{r['Country']}]" for r in results]
            selection = st.selectbox("Select Correct Stock:", options)
            if selection:
                ticker_to_analyze = selection.split(" | ")[0]
        else:
            st.warning("No matches found in database. You can try typing the exact ticker below.")
            ticker_to_analyze = st.text_input("Enter Exact Ticker Manually", value=user_query.upper())
    else:
        st.error("⚠️ Stock Database not loaded. Please ensure 'india_stocks.csv', 'us_stocks.csv', and 'canada_stocks.csv' are in the repository.")
        ticker_to_analyze = st.text_input("Enter Exact Ticker Manually", value=user_query.upper())


# --- ANALYSIS EXECUTION ---
if ticker_to_analyze:
    st.divider()
    st.markdown(f"### 2️⃣ Analyze: **{ticker_to_analyze}**")
    
    if st.button("🚀 Launch Strategic Deep Dive", use_container_width=True):
        if not api_key:
            st.error("⚠️ API Key Missing. Please add your Gemini API key in the sidebar.")
            st.stop()
        
        with st.spinner(f"Running Multi-Timeframe Analysis on {ticker_to_analyze}..."):
            
            # 1. Fetch Data
            stock, info, history_1h, history_daily, history_monthly = fetch_financial_data(ticker_to_analyze)
            if history_daily is None:
                st.error(f"❌ Could not load data for {ticker_to_analyze}. It might be delisted or the ticker format is incorrect.")
                st.stop()
            
            current_price = history_daily['Close'].iloc[-1]
            
            # 2. Run FVG Confluence Analysis
            fvg_results = analyze_fvg_confluence(ticker_to_analyze, history_1h, history_daily, history_monthly)

            # 3. AI Analysis
            company_name = info.get('shortName', '') or info.get('longName', '')
            
            fund_prompt = f"""
            Analyze {ticker_to_analyze} ({company_name}) Fundamentals:
            Price: {current_price} | PE: {info.get('trailingPE', 'N/A')} | PEG: {info.get('pegRatio', 'N/A')}
            Apply: Peter Lynch, Ben Graham. Verdict?
            """
            
            tech_prompt = f"""
            Analyze Technicals {ticker_to_analyze}. Last 30 Days: {history_daily.tail(30).to_csv()}
            Apply: Wyckoff, Price Action. Verdict?
            """
            
            social_data = get_social_sentiment_data(ticker_to_analyze, company_name)
            social_context = str(social_data[:5]) if social_data else "No data"
            
            soc_prompt = f"""
            Analyze Sentiment for {ticker_to_analyze}. News/Social: {social_context}
            Provide: Sentiment Score (1-10), Drivers, Risks.
            """
            
            dashboard_prompt = f"""
            Create JSON Dashboard for {ticker_to_analyze}.
            Price: {current_price}
            FVG Results: {fvg_results.get('bullish_confluence') if fvg_results else 'None'}
            Social: {social_context}
            
            Format JSON:
            {{
                "fundamentals_verdict": "string", "fundamentals_rating": int,
                "whats_great": "string", "whats_not_great": "string",
                "value_buy": "Yes/No", "risk_level": "Low/Med/High",
                "technicals_verdict": "string", "overall_trend": "Bull/Bear",
                "volume_analysis": "string", "support_analysis": "string",
                "resistance_analysis": "string", "chart_pattern": "string",
                "social_verdict": "string", "sentiment_score": int,
                "sentiment_outlook": "string", "sentiment_drivers": "string",
                "news_summary": "string", "social_pulse": "string", "sentiment_risks": "string"
            }}
            """
            
            # Generate Responses
            fund_an = get_gemini_response(fund_prompt)
            tech_an = get_gemini_response(tech_prompt)
            soc_an = get_gemini_response(soc_prompt)
            dashboard_raw = get_gemini_response(dashboard_prompt)
            
            try:
                clean_json = dashboard_raw.strip().replace('```json', '').replace('```', '')
                dashboard_data = json.loads(clean_json)
            except:
                dashboard_data = {}

            # --- DISPLAY RESULTS ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"{current_price:.2f} {info.get('currency', '')}")
            m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            m3.metric("🐂 Bull Confluences", len(fvg_results['bullish_confluence']) if fvg_results else 0)
            m4.metric("🐻 Bear Confluences", len(fvg_results['bearish_confluence']) if fvg_results else 0)
            
            t0, t1, t2, t3, t4 = st.tabs(["📊 Dashboard", "🎯 FVG Scanner", "🏛️ Fundamentals", "🔭 Technicals", "💬 Social"])
            
            with t0:
                st.subheader("Executive Dashboard")
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"**Fundamentals:** {dashboard_data.get('fundamentals_verdict', 'N/A')}")
                    st.info(f"**Technicals:** {dashboard_data.get('technicals_verdict', 'N/A')}")
                with c2:
                    st.warning(f"**Sentiment:** {dashboard_data.get('social_verdict', 'N/A')}")
                    st.write(f"**Risk Level:** {dashboard_data.get('risk_level', 'N/A')}")
            
            with t1:
                st.subheader("Multi-Timeframe FVG Confluence")
                if fvg_results:
                    if fvg_results['bullish_confluence']:
                        st.success("🐂 **Bullish Zones Detected**")
                        st.json(fvg_results['bullish_confluence'])
                    else: st.write("No Bullish Confluence")
                    
                    if fvg_results['bearish_confluence']:
                        st.error("🐻 **Bearish Zones Detected**")
                        st.json(fvg_results['bearish_confluence'])
                    else: st.write("No Bearish Confluence")

            with t2: st.markdown(fund_an)
            with t3: 
                fig = go.Figure(data=[go.Candlestick(x=history_daily.index, open=history_daily['Open'], high=history_daily['High'], low=history_daily['Low'], close=history_daily['Close'])])
                fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(tech_an)
            with t4: st.markdown(soc_an)