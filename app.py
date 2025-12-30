import streamlit as st
import yfinance as yf
import google.generativeai as genai
from duckduckgo_search import DDGS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import warnings
import time
import json

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
    region_selection = st.selectbox("Select Market Region", ["USA 🇺🇸", "India 🇮🇳", "Canada 🇨🇦"])
    region_map = {"USA 🇺🇸": "USA", "India 🇮🇳": "India", "Canada 🇨🇦": "Canada"}
    selected_country = region_map[region_selection]
    
    st.info("💡 **Ticker Guide:**\n- USA: `TSLA`, `AAPL`\n- India: `RELIANCE.NS`, `TCS.NS`\n- Canada: `SHOP.TO`, `RY.TO`")

# --- GEMINI API SETUP ---
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
    except Exception as e:
        st.sidebar.error(f"Model Error: {e}")

# --- MAIN HEADER ---
st.title("♟️ Company Analyzer")
st.markdown(f"*Strategic Deep Dive | Region: {selected_country}*")
st.markdown("---")

# --- SESSION STATE ---
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = None


# ==========================================
# FVG ENGINE - EXACT COPY FROM WORKING FVG SCANNER
# ==========================================

def calculate_bullish_fvgs(df):
    """
    Identifies ONLY Unmitigated Bullish FVGs (Support).
    
    STRICT RULES:
    1. Formation: FVG (Candles i, i+1, i+2) is valid only if i+2 is a CLOSED candle.
       We assume the last row of df (n-1) is the 'Current/Forming' candle.
       Therefore, i+2 must be <= n-2.
    
    2. Mitigation (UPDATED): 
       - An FVG is considered MITIGATED if any future CLOSED candle 'wicks' into the gap.
       - Specifically: If Future Low <= Top of FVG, it is mitigated.
       - We do not check the last row (current forming candle) for mitigation.
    """
    if df is None or len(df) < 4: 
        return []

    highs = df['High'].values
    lows = df['Low'].values
    
    n = len(lows)
    
    # Pre-compute Future Mitigation (Closed Candles Only)
    # min_future_low[k] = lowest Low of candles k, k+1, ... up to n-2
    # EXCLUDE n-1 (Current Candle) from this check
    min_future_low = np.full(n, np.inf)
    current_min = np.inf
    
    # Start from n-2 (Last Closed Candle) down to 0
    for i in range(n-2, -1, -1):
        current_min = min(current_min, lows[i])
        min_future_low[i] = current_min

    unmitigated_fvgs = []

    # Identify FVGs - trigger candle (i+2) must be closed (<= n-2)
    scan_limit = n - 3
    if scan_limit < 0:
        return []

    for i in range(scan_limit):
        # Bullish FVG Condition: High[i] < Low[i+2]
        if highs[i] < lows[i+2]:
            bottom = highs[i]
            top = lows[i+2]
            
            # Mitigation: If any future closed candle (i+3 to n-2) has Low <= Top
            if i + 3 <= n - 2:
                if min_future_low[i+3] <= top:
                    continue  # Mitigated (Wicked into)
            
            unmitigated_fvgs.append((bottom, top))

    return unmitigated_fvgs


def calculate_bearish_fvgs(df):
    """
    Identifies ONLY Unmitigated Bearish FVGs (Resistance).
    
    STRICT RULES:
    1. Formation: Bearish FVG when Low[i] > High[i+2]
    2. Mitigation: If any future closed candle wicks into gap (High >= Bottom)
    """
    if df is None or len(df) < 4: 
        return []

    highs = df['High'].values
    lows = df['Low'].values
    
    n = len(highs)
    
    # Pre-compute Future Mitigation (Closed Candles Only)
    max_future_high = np.full(n, -np.inf)
    current_max = -np.inf
    
    for i in range(n-2, -1, -1):
        current_max = max(current_max, highs[i])
        max_future_high[i] = current_max

    unmitigated_fvgs = []

    scan_limit = n - 3
    if scan_limit < 0:
        return []

    for i in range(scan_limit):
        # Bearish FVG Condition: Low[i] > High[i+2]
        if lows[i] > highs[i+2]:
            top = lows[i]
            bottom = highs[i+2]
            
            # Mitigation: If any future closed candle (i+3 to n-2) has High >= Bottom
            if i + 3 <= n - 2:
                if max_future_high[i+3] >= bottom:
                    continue  # Mitigated
            
            unmitigated_fvgs.append((bottom, top))

    return unmitigated_fvgs


def is_near_fvg(price, fvgs, threshold_pct):
    """
    Checks if price is within X% of any unmitigated FVG.
    Returns True if:
    1. Price is INSIDE the FVG, OR
    2. Price is within threshold_pct of nearest edge
    """
    if not fvgs:
        return False, None
    
    threshold = threshold_pct / 100.0
    
    for bot, top in fvgs:
        # 1. Check if price is INSIDE the FVG
        if bot <= price <= top:
            return True, {'bottom': bot, 'top': top, 'status': 'INSIDE'}
        
        # 2. Check distance to nearest edge
        dist_to_top = abs(price - top)
        dist_to_bot = abs(price - bot)
        min_dist = min(dist_to_top, dist_to_bot)
        
        if (min_dist / price) <= threshold:
            return True, {'bottom': bot, 'top': top, 'status': 'NEAR', 'distance_pct': (min_dist / price) * 100}
            
    return False, None


def resample_custom(df, timeframe):
    """
    Resamples data based on strict calendar rules - EXACT COPY FROM FVG SCANNER.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    if timeframe == '1D':
        return df.resample('1D').agg(agg_dict).dropna()
    
    if timeframe == '4H':
        return df.resample('4h').agg(agg_dict).dropna()

    if timeframe == '1W':
        return df.resample('W').agg(agg_dict).dropna()
    
    if timeframe == '1M':
        return df.resample('ME').agg(agg_dict).dropna()
        
    # Monthly based custom timeframes
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
# ANALYSIS ENGINE - USING CORRECT FVG LOGIC
# ==========================================

def analyze_fvg_confluence(ticker, df_1h, df_1d, df_1mo):
    """
    Analyzes FVG confluence across multiple timeframes using EXACT scanner logic.
    Returns detailed results for each timeframe pair.
    """
    if df_1d is None or df_1d.empty:
        return None
    
    current_price = df_1d['Close'].iloc[-1]
    
    # Prepare all timeframe dataframes
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
    
    # Calculate FVGs for each timeframe
    bullish_fvgs = {}
    bearish_fvgs = {}
    for tf, df in data_frames.items():
        if df is not None and not df.empty:
            bullish_fvgs[tf] = calculate_bullish_fvgs(df)
            bearish_fvgs[tf] = calculate_bearish_fvgs(df)
    
    # Define confluence pairs with thresholds (EXACT FROM SCANNER)
    confluence_pairs = [
        ('1H', '4H', 1.0, 1.0),
        ('1D', '1W', 2.0, 2.0),
        ('1W', '1M', 2.0, 3.0),
        ('1M', '3M', 3.0, 4.0),
        ('3M', '6M', 4.0, 5.0),
        ('6M', '12M', 5.0, 5.0),
    ]
    
    results = {
        'current_price': current_price,
        'bullish_confluence': {},
        'bearish_confluence': {},
        'bullish_fvg_counts': {},
        'bearish_fvg_counts': {},
    }
    
    # Store FVG counts
    for tf in data_frames.keys():
        results['bullish_fvg_counts'][tf] = len(bullish_fvgs.get(tf, []))
        results['bearish_fvg_counts'][tf] = len(bearish_fvgs.get(tf, []))
    
    # Check each confluence pair
    for tf1, tf2, thresh1, thresh2 in confluence_pairs:
        pair_name = f"{tf1}, {tf2}"
        
        # Bullish confluence
        if tf1 in bullish_fvgs and tf2 in bullish_fvgs:
            near1, zone1 = is_near_fvg(current_price, bullish_fvgs[tf1], thresh1)
            near2, zone2 = is_near_fvg(current_price, bullish_fvgs[tf2], thresh2)
            
            if near1 and near2:
                results['bullish_confluence'][pair_name] = {
                    'detected': True,
                    'tf1_zone': zone1,
                    'tf2_zone': zone2,
                    'thresholds': (thresh1, thresh2)
                }
        
        # Bearish confluence
        if tf1 in bearish_fvgs and tf2 in bearish_fvgs:
            near1, zone1 = is_near_fvg(current_price, bearish_fvgs[tf1], thresh1)
            near2, zone2 = is_near_fvg(current_price, bearish_fvgs[tf2], thresh2)
            
            if near1 and near2:
                results['bearish_confluence'][pair_name] = {
                    'detected': True,
                    'tf1_zone': zone1,
                    'tf2_zone': zone2,
                    'thresholds': (thresh1, thresh2)
                }
    
    return results


class MathWiz:
    """Additional technical analysis utilities."""
    
    @staticmethod
    def identify_strict_swings(df, neighbor_count=3):
        """Identify swing highs and swing lows."""
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
        """Check for Inverse FVG Reversal pattern."""
        subset = df.iloc[-5:].copy().dropna()
        if len(subset) < 5: return None

        c1 = subset.iloc[0]
        c3 = subset.iloc[2]
        c5 = subset.iloc[4]
        
        is_bear_fvg_first = c3['High'] < c1['Low']
        is_bull_fvg_second = c5['Low'] > c3['High']
        
        if is_bear_fvg_first and is_bull_fvg_second:
            return 'Bull'

        is_bull_fvg_first = c3['Low'] > c1['High']
        is_bear_fvg_second = c5['High'] < c3['Low']
        
        if is_bull_fvg_first and is_bear_fvg_second:
            return 'Bear'
            
        return None

    @staticmethod
    def check_consecutive_candles(df, num_candles):
        """Check for consecutive red or green candles."""
        if len(df) < num_candles:
            return None
        
        recent = df.iloc[-num_candles:]
        all_red = all(recent['Close'] < recent['Open'])
        all_green = all(recent['Close'] > recent['Open'])
        
        if all_red:
            return 'Bull'
        elif all_green:
            return 'Bear'
        
        return None

    @staticmethod
    def calculate_choppiness(high, low, close, length=14):
        """Calculate Choppiness Index."""
        try:
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr_sum = tr.rolling(window=length).sum()
            high_max = high.rolling(window=length).max()
            low_min = low.rolling(window=length).min()
            
            range_diff = high_max - low_min
            range_diff.replace(0, np.nan, inplace=True)

            numerator = np.log10(atr_sum / range_diff)
            denominator = np.log10(length)
            return 100 * (numerator / denominator)
        except:
            return pd.Series(dtype='float64')


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def search_tickers_safe(query, country):
    """Searches using DuckDuckGo with Ratelimit Handling."""
    candidates = []
    search_term = f"site:finance.yahoo.com/quote {query} {country} stock"
    try:
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

def get_gemini_response(prompt):
    """Generate content using the global model instance."""
    if model is None:
        return "**Error:** Model not initialized. Please provide API key."
    try:
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
    
    # Fetch all required timeframes
    history_1h = stock.history(period="730d", interval="1h")
    history_daily = stock.history(period="max", interval="1d")
    history_monthly = stock.history(period="max", interval="1mo")
    
    if history_daily.empty:
        return None, None, None, None, None
    return stock, info, history_1h, history_daily, history_monthly

def get_social_sentiment_data(ticker, company_name=""):
    """Fetches news and social mentions for sentiment analysis."""
    all_data = []
    
    try:
        time.sleep(0.5)
        with DDGS() as ddgs:
            try:
                news = list(ddgs.news(f"{ticker} stock", timelimit="w", max_results=5))
                for n in news:
                    all_data.append({
                        "type": "News",
                        "title": n.get('title', ''),
                        "source": n.get('source', 'Unknown'),
                        "date": n.get('date', ''),
                        "body": n.get('body', '')[:200] if n.get('body') else ''
                    })
            except:
                pass
            
            try:
                time.sleep(0.3)
                reddit_results = list(ddgs.text(f"{ticker} stock reddit discussion", max_results=3))
                for r in reddit_results:
                    if 'reddit' in r.get('href', '').lower():
                        all_data.append({
                            "type": "Reddit",
                            "title": r.get('title', ''),
                            "source": "Reddit",
                            "body": r.get('body', '')[:200] if r.get('body') else ''
                        })
            except:
                pass
            
            try:
                time.sleep(0.3)
                twitter_results = list(ddgs.text(f"{ticker} stock twitter OR x.com", max_results=3))
                for t in twitter_results:
                    if 'twitter' in t.get('href', '').lower() or 'x.com' in t.get('href', '').lower():
                        all_data.append({
                            "type": "Twitter/X",
                            "title": t.get('title', ''),
                            "source": "Twitter/X",
                            "body": t.get('body', '')[:200] if t.get('body') else ''
                        })
            except:
                pass
                
            try:
                time.sleep(0.3)
                general = list(ddgs.text(f"{ticker} stock buy sell hold analyst", max_results=3))
                for g in general:
                    all_data.append({
                        "type": "Analysis",
                        "title": g.get('title', ''),
                        "source": g.get('href', '').split('/')[2] if g.get('href') else 'Unknown',
                        "body": g.get('body', '')[:200] if g.get('body') else ''
                    })
            except:
                pass
                
    except Exception as e:
        pass
    
    return all_data


# ==========================================
# UI WORKFLOW
# ==========================================
st.markdown("### 1️⃣ Identify Target")

col_mode, col_input = st.columns([1, 3])

with col_mode:
    input_mode = st.radio("Input Method:", ["Search by Name", "Direct Ticker Input"])

ticker_to_analyze = None

with col_input:
    if input_mode == "Search by Name":
        c_search_1, c_search_2 = st.columns([3, 1])
        with c_search_1:
            search_query = st.text_input("Company Name", placeholder="e.g. Tesla, Reliance...")
        with c_search_2:
            st.write("")
            st.write("") 
            if st.button("🔎 Find", use_container_width=True):
                if search_query:
                    with st.spinner("Scanning..."):
                        results = search_tickers_safe(search_query, selected_country)
                        st.session_state.search_results = results
        
        if st.session_state.search_results:
            options = [f"{item['symbol']} | {item['title']}" for item in st.session_state.search_results]
            selected_option = st.selectbox("Select Correct Stock:", options)
            if selected_option:
                ticker_to_analyze = selected_option.split(" | ")[0]
        elif search_query and not st.session_state.search_results:
             st.warning("Search returned no results or was blocked. Please switch to 'Direct Ticker Input'.")
             
    else:
        direct_input = st.text_input("Enter Exact Ticker Symbol", placeholder="e.g. TSLA, RELIANCE.NS, SHOP.TO")
        if direct_input:
            ticker_to_analyze = direct_input.strip().upper()

# --- ANALYSIS EXECUTION ---
if ticker_to_analyze:
    st.divider()
    st.markdown(f"### 2️⃣ Analyze: **{ticker_to_analyze}**")
    
    if st.button("🚀 Launch Strategic Deep Dive", use_container_width=True):
        if not api_key:
            st.error("⚠️ API Key Missing. Please add your Gemini API key in the sidebar.")
            st.stop()
        
        if model is None:
            st.error("⚠️ Model not initialized. Check your API key.")
            st.stop()
            
        with st.spinner(f"Running Multi-Timeframe Analysis on {ticker_to_analyze}..."):
            
            # 1. Fetch Data
            stock, info, history_1h, history_daily, history_monthly = fetch_financial_data(ticker_to_analyze)
            if history_daily is None:
                st.error(f"❌ Could not load data for {ticker_to_analyze}. Check if ticker is correct for Region: {selected_country}.")
                st.stop()
            
            current_price = history_daily['Close'].iloc[-1]
            
            # 2. Run FVG Confluence Analysis (EXACT SCANNER LOGIC)
            fvg_results = analyze_fvg_confluence(ticker_to_analyze, history_1h, history_daily, history_monthly)

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
            Last 30 Days OHLCV: {history_daily.tail(30).to_csv()}
            Apply: Murphy, Wyckoff, Nison, Brooks.
            Verdict: Bullish/Bearish/Neutral?
            """
            
            company_name = info.get('shortName', '') or info.get('longName', '')
            social_data = get_social_sentiment_data(ticker_to_analyze, company_name)
            
            if social_data:
                social_context = "\n".join([
                    f"[{item['type']}] {item['title']} - {item.get('body', '')}" 
                    for item in social_data[:10]
                ])
            else:
                social_context = "Limited social data available"
            
            soc_prompt = f"""
            You are a Senior Market Sentiment Analyst. Analyze the social sentiment for {ticker_to_analyze} ({company_name}).
            
            Current Price: ${current_price:.2f}
            P/E Ratio: {info.get('trailingPE', 'N/A')}
            
            RECENT NEWS & SOCIAL DATA:
            {social_context}
            
            Provide a comprehensive sentiment analysis in this EXACT format:
            
            ## 📊 Overall Sentiment Score
            [Give a score from 1-10 and a one-word verdict: Very Bearish / Bearish / Neutral / Bullish / Very Bullish]
            
            ## 🔥 Key Sentiment Drivers
            [List 3-4 main factors driving current sentiment - positive and negative]
            
            ## 📰 News Sentiment Summary  
            [Summarize what the news is saying - bullish or bearish catalysts]
            
            ## 💬 Social Media Pulse (Reddit/X)
            [Summarize retail investor sentiment - what are retail traders saying/feeling]
            
            ## ⚠️ Sentiment Risks
            [Any contrarian signals or risks to watch]
            
            ## 🎯 Sentiment-Based Outlook
            [1-2 sentence actionable insight based on sentiment]
            
            Be specific, data-driven, and avoid generic statements.
            """
            
            fund_an = get_gemini_response(fund_prompt)
            tech_an = get_gemini_response(tech_prompt)
            soc_an = get_gemini_response(soc_prompt)
            
            # 4. Generate Dashboard Summary (NEW)
            # Prepare FVG confluence info for dashboard
            bullish_conf_pairs = list(fvg_results['bullish_confluence'].keys()) if fvg_results else []
            bearish_conf_pairs = list(fvg_results['bearish_confluence'].keys()) if fvg_results else []
            
            fvg_confluence_str = ', '.join(bullish_conf_pairs) if bullish_conf_pairs else "None"
            bearish_confluence_str = ', '.join(bearish_conf_pairs) if bearish_conf_pairs else "None"
            
            # Get FVG counts
            bull_fvg_counts = fvg_results.get('bullish_fvg_counts', {}) if fvg_results else {}
            bear_fvg_counts = fvg_results.get('bearish_fvg_counts', {}) if fvg_results else {}
            
            # Calculate additional signals (iFVG, RevCand, Squeeze, etc.)
            d_1d = resample_custom(history_daily, '1D')
            d_1w = resample_custom(history_daily, '1W')
            
            squeeze_1d = False
            squeeze_1w = False
            exhaustion = False
            
            if not d_1d.empty:
                chop_d = MathWiz.calculate_choppiness(d_1d['High'], d_1d['Low'], d_1d['Close'])
                if not chop_d.empty and not pd.isna(chop_d.iloc[-1]):
                    squeeze_1d = chop_d.iloc[-1] > 59
                    
            if not d_1w.empty:
                chop_w = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
                if not chop_w.empty and not pd.isna(chop_w.iloc[-1]):
                    squeeze_1w = chop_w.iloc[-1] > 59
                    exhaustion = chop_w.iloc[-1] < 25
            
            # Check for reversal candidates
            rev_cand_1d = MathWiz.check_consecutive_candles(d_1d, 5) if not d_1d.empty and len(d_1d) >= 5 else None
            rev_cand_1w = MathWiz.check_consecutive_candles(d_1w, 4) if not d_1w.empty and len(d_1w) >= 4 else None
            
            # Check for iFVG
            ifvg_1d = MathWiz.check_ifvg_reversal(d_1d) if not d_1d.empty else None
            ifvg_1w = MathWiz.check_ifvg_reversal(d_1w) if not d_1w.empty else None
            
            dashboard_prompt = f"""
            You are an elite equity research analyst. Create a PRECISE executive dashboard summary for {ticker_to_analyze}.
            
            STOCK DATA:
            - Current Price: ${current_price:.2f}
            - Market Cap: {info.get('marketCap', 'N/A')}
            - P/E: {info.get('trailingPE', 'N/A')}
            - PEG: {info.get('pegRatio', 'N/A')}
            - P/B: {info.get('priceToBook', 'N/A')}
            - 52W High: {info.get('fiftyTwoWeekHigh', 'N/A')}
            - 52W Low: {info.get('fiftyTwoWeekLow', 'N/A')}
            - Analyst Recommendation: {info.get('recommendationKey', 'N/A')}
            
            TECHNICAL DATA (Last 30 days):
            {history_daily.tail(30).to_csv()}
            
            FVG SCANNER RESULTS (Using Strict Unmitigated FVG Logic):
            - Bullish FVG Confluence Pairs: {fvg_confluence_str}
            - Bearish FVG Confluence Pairs: {bearish_confluence_str}
            - Bullish FVG Counts by TF: {bull_fvg_counts}
            - Bearish FVG Counts by TF: {bear_fvg_counts}
            - Daily Squeeze (Chop>59): {'Yes' if squeeze_1d else 'No'}
            - Weekly Squeeze (Chop>59): {'Yes' if squeeze_1w else 'No'}
            - Exhaustion (Chop<25): {'Yes' if exhaustion else 'No'}
            - Daily Reversal Candidate: {rev_cand_1d or 'None'}
            - Weekly Reversal Candidate: {rev_cand_1w or 'None'}
            - Daily iFVG: {ifvg_1d or 'None'}
            - Weekly iFVG: {ifvg_1w or 'None'}
            
            NEWS & SOCIAL DATA:
            {social_context}
            
            Respond in EXACTLY this JSON format (no markdown, no code blocks, just raw JSON):
            {{
                "fundamentals_verdict": "One sentence verdict on fundamentals",
                "fundamentals_rating": 7,
                "whats_great": "One sentence on key strengths",
                "whats_not_great": "One sentence on key weaknesses",
                "value_buy": "Yes/No with brief reason",
                "risk_level": "Low/Medium/High",
                "technicals_verdict": "One sentence on technical setup",
                "overall_trend": "Bullish/Bearish/Neutral with brief context",
                "volume_analysis": "One sentence on volume patterns",
                "support_analysis": "Key support level and strength",
                "resistance_analysis": "Key resistance level and strength",
                "chart_pattern": "Current pattern if any",
                "fvg_confluence": "TFs with confluence or None",
                "social_verdict": "One sentence sentiment verdict",
                "sentiment_score": 7,
                "sentiment_outlook": "One sentence actionable outlook",
                "sentiment_drivers": "Top 2-3 drivers in one sentence",
                "news_summary": "One sentence news sentiment",
                "social_pulse": "One sentence Reddit/X sentiment",
                "sentiment_risks": "Key contrarian risk in one sentence"
            }}
            
            Be extremely precise and data-driven. No fluff. Every field must be filled.
            """
            
            dashboard_raw = get_gemini_response(dashboard_prompt)
            
            # Parse dashboard JSON
            try:
                # Clean up response - remove markdown code blocks if present
                clean_json = dashboard_raw.strip()
                if clean_json.startswith("```"):
                    clean_json = clean_json.split("```")[1]
                    if clean_json.startswith("json"):
                        clean_json = clean_json[4:]
                clean_json = clean_json.strip()
                dashboard_data = json.loads(clean_json)
            except:
                # Fallback if JSON parsing fails
                dashboard_data = {
                    "fundamentals_verdict": "Analysis pending - see Fundamentals tab",
                    "fundamentals_rating": "N/A",
                    "whats_great": "See detailed analysis",
                    "whats_not_great": "See detailed analysis",
                    "value_buy": "See detailed analysis",
                    "risk_level": "Medium",
                    "technicals_verdict": "Analysis pending - see Technicals tab",
                    "overall_trend": "See chart analysis",
                    "volume_analysis": "See detailed analysis",
                    "support_analysis": "See detailed analysis",
                    "resistance_analysis": "See detailed analysis",
                    "chart_pattern": "See chart",
                    "fvg_confluence": fvg_confluence_str,
                    "social_verdict": "Analysis pending - see Social tab",
                    "sentiment_score": "N/A",
                    "sentiment_outlook": "See detailed analysis",
                    "sentiment_drivers": "See detailed analysis",
                    "news_summary": "See detailed analysis",
                    "social_pulse": "See detailed analysis",
                    "sentiment_risks": "See detailed analysis"
                }

            # --- DISPLAY RESULTS ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"{current_price:.2f} {info.get('currency', '')}")
            m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            m3.metric("🐂 Bullish Confluences", len(bullish_conf_pairs))
            m4.metric("🐻 Bearish Confluences", len(bearish_conf_pairs))
            
            t0, t1, t2, t3, t4 = st.tabs(["📊 Dashboard", "🎯 FVG Scanner", "🏛️ Fundamentals", "🔭 Technicals", "💬 Social"])
            
            with t0:
                st.markdown("### 📊 Executive Dashboard")
                st.caption(f"AI-Powered Summary for {ticker_to_analyze} ({company_name})")
                
                # --- FUNDAMENTALS SECTION ---
                st.markdown("#### 🏛️ Fundamentals")
                
                fund_col1, fund_col2 = st.columns([3, 1])
                with fund_col1:
                    st.markdown(f"**Verdict:** {dashboard_data.get('fundamentals_verdict', 'N/A')}")
                with fund_col2:
                    rating = dashboard_data.get('fundamentals_rating', 'N/A')
                    if isinstance(rating, (int, float)):
                        color = "🟢" if rating >= 7 else "🟡" if rating >= 5 else "🔴"
                        st.markdown(f"**Rating:** {color} {rating}/10")
                    else:
                        st.markdown(f"**Rating:** {rating}/10")
                
                f1, f2 = st.columns(2)
                with f1:
                    st.success(f"✅ **What's Great:** {dashboard_data.get('whats_great', 'N/A')}")
                with f2:
                    st.error(f"⚠️ **What's Not Great:** {dashboard_data.get('whats_not_great', 'N/A')}")
                
                f3, f4 = st.columns(2)
                with f3:
                    value_buy = dashboard_data.get('value_buy', 'N/A')
                    if 'yes' in str(value_buy).lower():
                        st.info(f"💰 **Value Buy:** {value_buy}")
                    else:
                        st.warning(f"💰 **Value Buy:** {value_buy}")
                with f4:
                    risk = dashboard_data.get('risk_level', 'Medium')
                    risk_color = "🟢" if 'low' in str(risk).lower() else "🔴" if 'high' in str(risk).lower() else "🟡"
                    st.markdown(f"**Risk Level:** {risk_color} {risk}")
                
                st.divider()
                
                # --- TECHNICALS SECTION ---
                st.markdown("#### 🔭 Technicals")
                
                st.markdown(f"**Verdict:** {dashboard_data.get('technicals_verdict', 'N/A')}")
                
                tech_row1 = st.columns(3)
                with tech_row1[0]:
                    trend = dashboard_data.get('overall_trend', 'N/A')
                    trend_icon = "📈" if 'bullish' in str(trend).lower() else "📉" if 'bearish' in str(trend).lower() else "➡️"
                    st.markdown(f"**{trend_icon} Trend:** {trend}")
                with tech_row1[1]:
                    st.markdown(f"**📊 Volume:** {dashboard_data.get('volume_analysis', 'N/A')}")
                with tech_row1[2]:
                    st.markdown(f"**📐 Pattern:** {dashboard_data.get('chart_pattern', 'N/A')}")
                
                tech_row2 = st.columns(3)
                with tech_row2[0]:
                    st.markdown(f"**🟢 Support:** {dashboard_data.get('support_analysis', 'N/A')}")
                with tech_row2[1]:
                    st.markdown(f"**🔴 Resistance:** {dashboard_data.get('resistance_analysis', 'N/A')}")
                with tech_row2[2]:
                    fvg_conf = dashboard_data.get('fvg_confluence', fvg_confluence_str)
                    if fvg_conf and fvg_conf != "None":
                        st.success(f"**🎯 FVG Confluence:** {fvg_conf}")
                    else:
                        st.markdown(f"**🎯 FVG Confluence:** None")
                
                st.divider()
                
                # --- SENTIMENT SECTION ---
                st.markdown("#### 💬 Sentiment")
                
                sent_col1, sent_col2 = st.columns([3, 1])
                with sent_col1:
                    st.markdown(f"**Verdict:** {dashboard_data.get('social_verdict', 'N/A')}")
                with sent_col2:
                    sent_score = dashboard_data.get('sentiment_score', 'N/A')
                    if isinstance(sent_score, (int, float)):
                        sent_color = "🟢" if sent_score >= 7 else "🟡" if sent_score >= 5 else "🔴"
                        st.markdown(f"**Score:** {sent_color} {sent_score}/10")
                    else:
                        st.markdown(f"**Score:** {sent_score}/10")
                
                st.info(f"🎯 **Outlook:** {dashboard_data.get('sentiment_outlook', 'N/A')}")
                
                sent_row1 = st.columns(2)
                with sent_row1[0]:
                    st.markdown(f"**🔥 Key Drivers:** {dashboard_data.get('sentiment_drivers', 'N/A')}")
                with sent_row1[1]:
                    st.markdown(f"**📰 News:** {dashboard_data.get('news_summary', 'N/A')}")
                
                sent_row2 = st.columns(2)
                with sent_row2[0]:
                    st.markdown(f"**💬 Social Pulse:** {dashboard_data.get('social_pulse', 'N/A')}")
                with sent_row2[1]:
                    st.warning(f"**⚠️ Risks:** {dashboard_data.get('sentiment_risks', 'N/A')}")
                
                st.divider()
                
                # --- QUICK STATS ---
                st.markdown("#### 📈 Quick Stats")
                qs1, qs2, qs3, qs4, qs5 = st.columns(5)
                qs1.metric("Price", f"${current_price:.2f}")
                qs2.metric("P/E", f"{info.get('trailingPE', 'N/A')}")
                qs3.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                qs4.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
                qs5.metric("Analyst", f"{info.get('recommendationKey', 'N/A').upper() if info.get('recommendationKey') else 'N/A'}")
            
            with t1:
                st.markdown("### 🎯 Multi-Timeframe FVG Confluence Scanner")
                st.caption("Using strict unmitigated FVG logic: Current candle ignored, mitigation = wick into gap")
                
                # --- CONFLUENCE CRITERIA ---
                st.markdown("""
                **Confluence Criteria (Price must be near FVG on BOTH timeframes):**
                | Pair | TF1 Threshold | TF2 Threshold |
                |------|---------------|---------------|
                | 1H, 4H | 1% | 1% |
                | 1D, 1W | 2% | 2% |
                | 1W, 1M | 2% | 3% |
                | 1M, 3M | 3% | 4% |
                | 3M, 6M | 4% | 5% |
                | 6M, 12M | 5% | 5% |
                """)
                
                st.divider()
                
                # --- BULLISH CONFLUENCE RESULTS ---
                st.markdown("#### 🐂 Bullish FVG Confluence (Support Zones)")
                
                if fvg_results and fvg_results['bullish_confluence']:
                    for pair_name, data in fvg_results['bullish_confluence'].items():
                        with st.expander(f"✅ {pair_name} - CONFLUENCE DETECTED", expanded=True):
                            col1, col2 = st.columns(2)
                            tf1, tf2 = pair_name.split(', ')
                            with col1:
                                st.markdown(f"**{tf1} Zone:**")
                                zone1 = data['tf1_zone']
                                st.write(f"- Bottom: ${zone1['bottom']:.2f}")
                                st.write(f"- Top: ${zone1['top']:.2f}")
                                st.write(f"- Status: {zone1['status']}")
                            with col2:
                                st.markdown(f"**{tf2} Zone:**")
                                zone2 = data['tf2_zone']
                                st.write(f"- Bottom: ${zone2['bottom']:.2f}")
                                st.write(f"- Top: ${zone2['top']:.2f}")
                                st.write(f"- Status: {zone2['status']}")
                            st.info(f"Thresholds: {tf1}={data['thresholds'][0]}%, {tf2}={data['thresholds'][1]}%")
                else:
                    st.info("❌ No Bullish FVG Confluence detected at current price level.")
                
                # --- FVG COUNTS BY TIMEFRAME ---
                st.markdown("#### 📊 Unmitigated Bullish FVG Count by Timeframe")
                if fvg_results:
                    fvg_df = pd.DataFrame([{
                        'Timeframe': tf,
                        'Bullish FVGs': count
                    } for tf, count in fvg_results['bullish_fvg_counts'].items()])
                    st.dataframe(fvg_df, hide_index=True, use_container_width=True)
                
                st.divider()
                
                # --- BEARISH CONFLUENCE RESULTS ---
                st.markdown("#### 🐻 Bearish FVG Confluence (Resistance Zones)")
                
                if fvg_results and fvg_results['bearish_confluence']:
                    for pair_name, data in fvg_results['bearish_confluence'].items():
                        with st.expander(f"✅ {pair_name} - CONFLUENCE DETECTED", expanded=True):
                            col1, col2 = st.columns(2)
                            tf1, tf2 = pair_name.split(', ')
                            with col1:
                                st.markdown(f"**{tf1} Zone:**")
                                zone1 = data['tf1_zone']
                                st.write(f"- Bottom: ${zone1['bottom']:.2f}")
                                st.write(f"- Top: ${zone1['top']:.2f}")
                                st.write(f"- Status: {zone1['status']}")
                            with col2:
                                st.markdown(f"**{tf2} Zone:**")
                                zone2 = data['tf2_zone']
                                st.write(f"- Bottom: ${zone2['bottom']:.2f}")
                                st.write(f"- Top: ${zone2['top']:.2f}")
                                st.write(f"- Status: {zone2['status']}")
                            st.info(f"Thresholds: {tf1}={data['thresholds'][0]}%, {tf2}={data['thresholds'][1]}%")
                else:
                    st.info("❌ No Bearish FVG Confluence detected at current price level.")
                
                # --- BEARISH FVG COUNTS ---
                st.markdown("#### 📊 Unmitigated Bearish FVG Count by Timeframe")
                if fvg_results:
                    bear_fvg_df = pd.DataFrame([{
                        'Timeframe': tf,
                        'Bearish FVGs': count
                    } for tf, count in fvg_results['bearish_fvg_counts'].items()])
                    st.dataframe(bear_fvg_df, hide_index=True, use_container_width=True)
                
                st.divider()
                
                # --- ADDITIONAL SIGNALS ---
                st.markdown("#### 🔍 Additional Technical Signals")
                
                sig_col1, sig_col2, sig_col3 = st.columns(3)
                
                with sig_col1:
                    st.markdown("**Squeeze (Chop > 59)**")
                    if squeeze_1d:
                        st.success("✅ Daily Squeeze Active")
                    else:
                        st.info("❌ No Daily Squeeze")
                    if squeeze_1w:
                        st.success("✅ Weekly Squeeze Active")
                    else:
                        st.info("❌ No Weekly Squeeze")
                
                with sig_col2:
                    st.markdown("**Reversal Candidates**")
                    if rev_cand_1d == 'Bull':
                        st.success("✅ Daily: 5 Red Candles (Bullish Rev)")
                    elif rev_cand_1d == 'Bear':
                        st.error("⚠️ Daily: 5 Green Candles (Bearish Rev)")
                    else:
                        st.info("❌ No Daily Reversal Signal")
                    
                    if rev_cand_1w == 'Bull':
                        st.success("✅ Weekly: 4 Red Candles (Bullish Rev)")
                    elif rev_cand_1w == 'Bear':
                        st.error("⚠️ Weekly: 4 Green Candles (Bearish Rev)")
                    else:
                        st.info("❌ No Weekly Reversal Signal")
                
                with sig_col3:
                    st.markdown("**iFVG & Exhaustion**")
                    if ifvg_1d == 'Bull':
                        st.success("✅ Daily Bullish iFVG")
                    elif ifvg_1d == 'Bear':
                        st.error("⚠️ Daily Bearish iFVG")
                    else:
                        st.info("❌ No Daily iFVG")
                    
                    if exhaustion:
                        st.error("⚠️ **EXHAUSTION** (Weekly Chop < 25)")
                    else:
                        st.info("❌ No Exhaustion Signal")
            
            with t2: 
                st.markdown(fund_an)
            
            with t3:
                fig = go.Figure(data=[go.Candlestick(
                    x=history_daily.index, 
                    open=history_daily['Open'], 
                    high=history_daily['High'], 
                    low=history_daily['Low'], 
                    close=history_daily['Close']
                )])
                fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(tech_an)
            
            with t4: 
                st.markdown(soc_an)
                
                if social_data:
                    st.markdown("---")
                    with st.expander("📚 View Raw Data Sources", expanded=False):
                        for item in social_data:
                            source_icon = "📰" if item['type'] == "News" else "🤖" if item['type'] == "Reddit" else "🐦" if item['type'] == "Twitter/X" else "📊"
                            st.markdown(f"**{source_icon} [{item['type']}]** {item['title']}")
                            if item.get('source'):
                                st.caption(f"Source: {item['source']}")
                            st.markdown("---")