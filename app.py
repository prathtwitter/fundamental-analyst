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
# MATH WIZ - EXACT COPY FROM WORKING SCANNER
# ==========================================
class MathWiz:
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
    def find_fvg(df):
        """Find Fair Value Gaps - EXACT LOGIC FROM SCANNER."""
        bull_fvg = (df['Low'] > df['High'].shift(2))
        bear_fvg = (df['High'] < df['Low'].shift(2))
        return bull_fvg, bear_fvg
    
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
    def find_unmitigated_fvg_zone(df, threshold_pct=0.05):
        """Check if price is near an unmitigated FVG zone - EXACT LOGIC FROM SCANNER."""
        if len(df) < 5: return False, None
        current_price = df['Close'].iloc[-1]
        lookback = min(len(df), 50)
        
        for i in range(len(df)-1, max(len(df)-lookback, 2), -1):
            curr_low = df['Low'].iloc[i]
            prev_high = df['High'].iloc[i-2]
            
            # Bullish FVG: Current candle's low > candle 2 bars ago high
            if curr_low > prev_high: 
                gap_top = curr_low
                gap_bottom = prev_high
                
                # Check if gap has been mitigated (price went below gap_bottom)
                subsequent_data = df.iloc[i+1:]
                if not subsequent_data.empty:
                    if (subsequent_data['Low'] < gap_bottom).any():
                        continue  # Gap was mitigated, skip
                
                # Check if current price is within threshold of gap bottom
                upper_bound = gap_bottom * (1 + threshold_pct)
                lower_bound = gap_bottom * (1 - threshold_pct)
                if lower_bound <= current_price <= upper_bound:
                    return True, {
                        'date': df.index[i],
                        'gap_top': gap_top,
                        'gap_bottom': gap_bottom,
                        'current_price': current_price
                    }
        return False, None

    @staticmethod
    def check_consecutive_candles(df, num_candles):
        """Check for consecutive red or green candles."""
        if len(df) < num_candles:
            return None
        
        recent = df.iloc[-num_candles:]
        all_red = all(recent['Close'] < recent['Open'])
        all_green = all(recent['Close'] > recent['Open'])
        
        if all_red:
            return 'Bull'  # Reversal candidate
        elif all_green:
            return 'Bear'  # Reversal candidate
        
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
# DATA ENGINE - EXACT COPY FROM SCANNER
# ==========================================
def resample_custom(df, timeframe):
    """Resample data to different timeframes - EXACT LOGIC FROM SCANNER."""
    if df.empty: return df
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    try:
        if timeframe == "1D": return df.resample("1D").agg(agg_dict).dropna()
        if timeframe == "1W": return df.resample("W-FRI").agg(agg_dict).dropna()
        if timeframe == "1M": return df.resample("ME").agg(agg_dict).dropna()

        df_monthly = df.resample('MS').agg(agg_dict).dropna()
        if timeframe == "3M": return df_monthly.resample('QE').agg(agg_dict).dropna()
        if timeframe == "6M":
            df_monthly['Year'] = df_monthly.index.year
            df_monthly['Half'] = np.where(df_monthly.index.month <= 6, 1, 2)
            df_6m = df_monthly.groupby(['Year', 'Half']).agg(agg_dict)
            new_index = []
            for (year, half) in df_6m.index:
                month = 6 if half == 1 else 12
                new_index.append(pd.Timestamp(year=year, month=month, day=30))
            df_6m.index = pd.DatetimeIndex(new_index)
            return df_6m.sort_index()
        if timeframe == "12M": return df_monthly.resample('YE').agg(agg_dict).dropna()
    except: return df
    return df


# ==========================================
# ANALYSIS ENGINE - EXACT LOGIC FROM SCANNER
# ==========================================
def analyze_single_ticker(ticker, df_daily_raw, df_monthly_raw):
    """Runs ALL scans for ONE ticker - EXACT LOGIC FROM SCANNER."""
    results = {
        'Ticker': ticker,
        'Price': 0,
        # Bullish signals
        'Bull_OB_1D': False, 'Bull_OB_1W': False, 'Bull_OB_1M': False,
        'Bull_FVG_1D': False, 'Bull_FVG_1W': False, 'Bull_FVG_1M': False,
        'Bull_RevCand_1D': False, 'Bull_RevCand_1W': False, 'Bull_RevCand_1M': False,
        'Bull_iFVG_1D': False, 'Bull_iFVG_1W': False, 'Bull_iFVG_1M': False,
        'Support_3M': False, 'Support_6M': False, 'Support_12M': False,
        'Squeeze_1D': False, 'Squeeze_1W': False,
        # Bearish signals
        'Bear_OB_1D': False, 'Bear_OB_1W': False, 'Bear_OB_1M': False,
        'Bear_FVG_1D': False, 'Bear_FVG_1W': False, 'Bear_FVG_1M': False,
        'Bear_RevCand_1D': False, 'Bear_RevCand_1W': False, 'Bear_RevCand_1M': False,
        'Bear_iFVG_1D': False, 'Bear_iFVG_1W': False, 'Bear_iFVG_1M': False,
        'Exhaustion': False,
        # Confluence tracking
        'support_zones': [],
        'bullish_count': 0,
        'bearish_count': 0
    }
    
    try:
        d_1d = resample_custom(df_daily_raw, "1D")
        d_1w = resample_custom(df_daily_raw, "1W")
        d_1m = resample_custom(df_monthly_raw, "1M")
        d_3m = resample_custom(df_monthly_raw, "3M")
        d_6m = resample_custom(df_monthly_raw, "6M")
        d_12m = resample_custom(df_monthly_raw, "12M")
        
        data_map = {"1D": d_1d, "1W": d_1w, "1M": d_1m, "3M": d_3m, "6M": d_6m, "12M": d_12m}
        
        if not d_1d.empty:
            results['Price'] = round(d_1d['Close'].iloc[-1], 2)
        
    except: 
        return results

    # --- SCAN EACH TIMEFRAME (1D, 1W, 1M) ---
    for tf in ["1D", "1W", "1M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        
        df = data_map[tf].copy() 
        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_Swing_High'], df['Is_Swing_Low'] = MathWiz.identify_strict_swings(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # --- BULLISH FVG + Swing Break ---
        if curr['Bull_FVG']:
            past_swings = df[df['Is_Swing_High']]
            if not past_swings.empty:
                last_swing_high = past_swings['High'].iloc[-1]
                if curr['Close'] > last_swing_high and prev['Close'] <= last_swing_high:
                    results[f'Bull_FVG_{tf}'] = True

        # --- BEARISH FVG + Swing Break ---
        if curr['Bear_FVG']:
            past_swings = df[df['Is_Swing_Low']]
            if not past_swings.empty:
                last_swing_low = past_swings['Low'].iloc[-1]
                if curr['Close'] < last_swing_low and prev['Close'] >= last_swing_low:
                    results[f'Bear_FVG_{tf}'] = True

        # --- BULLISH ORDER BLOCK: 1 bearish + 3 bullish + FVG ---
        subset = df.iloc[-4:].copy()
        if len(subset) == 4:
            c1, c2, c3, c4 = subset.iloc[0], subset.iloc[1], subset.iloc[2], subset.iloc[3]
            c1_bearish = c1['Close'] < c1['Open']
            c2_bullish = c2['Close'] > c2['Open']
            c3_bullish = c3['Close'] > c3['Open']
            c4_bullish = c4['Close'] > c4['Open']
            c4_has_bull_fvg = c4['Low'] > c2['High']
            
            if c1_bearish and c2_bullish and c3_bullish and c4_bullish and c4_has_bull_fvg:
                results[f'Bull_OB_{tf}'] = True

        # --- BEARISH ORDER BLOCK: 1 bullish + 3 bearish + FVG ---
        if len(subset) == 4:
            c1_bullish = c1['Close'] > c1['Open']
            c2_bearish = c2['Close'] < c2['Open']
            c3_bearish = c3['Close'] < c3['Open']
            c4_bearish = c4['Close'] < c4['Open']
            c4_has_bear_fvg = c4['High'] < c2['Low']
            
            if c1_bullish and c2_bearish and c3_bearish and c4_bearish and c4_has_bear_fvg:
                results[f'Bear_OB_{tf}'] = True
        
        # --- iFVG Reversal ---
        ifvg_status = MathWiz.check_ifvg_reversal(df)
        if ifvg_status == "Bull":
            results[f'Bull_iFVG_{tf}'] = True
        elif ifvg_status == "Bear":
            results[f'Bear_iFVG_{tf}'] = True

    # --- SUPPORT ZONES (Higher TFs: 3M, 6M, 12M) ---
    support_zones = []
    for tf_name, tf_df in [("3M", d_3m), ("6M", d_6m), ("12M", d_12m)]:
        found, zone_data = MathWiz.find_unmitigated_fvg_zone(tf_df)
        if found:
            results[f'Support_{tf_name}'] = True
            support_zones.append({'timeframe': tf_name, 'zone': zone_data})
    
    results['support_zones'] = support_zones

    # --- SQUEEZE (Choppiness > 59) ---
    if not d_1d.empty:
        chop_series_d = MathWiz.calculate_choppiness(d_1d['High'], d_1d['Low'], d_1d['Close'])
        if not chop_series_d.empty and not pd.isna(chop_series_d.iloc[-1]):
            if chop_series_d.iloc[-1] > 59:
                results['Squeeze_1D'] = True
                
    if not d_1w.empty:
        chop_series_w = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series_w.empty and not pd.isna(chop_series_w.iloc[-1]):
            if chop_series_w.iloc[-1] > 59:
                results['Squeeze_1W'] = True

    # --- EXHAUSTION (Choppiness < 25 on Weekly) ---
    if not d_1w.empty:
        chop_series = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series.empty and not pd.isna(chop_series.iloc[-1]):
            if chop_series.iloc[-1] < 25:
                results['Exhaustion'] = True

    # --- REVERSAL CANDIDATES (Consecutive Candles) ---
    if not d_1d.empty and len(d_1d) >= 5:
        rev = MathWiz.check_consecutive_candles(d_1d, 5)
        if rev == 'Bull': results['Bull_RevCand_1D'] = True
        elif rev == 'Bear': results['Bear_RevCand_1D'] = True
    
    if not d_1w.empty and len(d_1w) >= 4:
        rev = MathWiz.check_consecutive_candles(d_1w, 4)
        if rev == 'Bull': results['Bull_RevCand_1W'] = True
        elif rev == 'Bear': results['Bear_RevCand_1W'] = True
    
    if not d_1m.empty and len(d_1m) >= 3:
        rev = MathWiz.check_consecutive_candles(d_1m, 3)
        if rev == 'Bull': results['Bull_RevCand_1M'] = True
        elif rev == 'Bear': results['Bear_RevCand_1M'] = True

    # --- COUNT SIGNALS ---
    bull_keys = [k for k in results.keys() if k.startswith('Bull_') or k.startswith('Support_') or k.startswith('Squeeze_')]
    bear_keys = [k for k in results.keys() if k.startswith('Bear_') or k == 'Exhaustion']
    
    results['bullish_count'] = sum(1 for k in bull_keys if results.get(k, False))
    results['bearish_count'] = sum(1 for k in bear_keys if results.get(k, False))

    return results


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
    history_daily = stock.history(period="2y", interval="1d")
    history_monthly = stock.history(period="max", interval="1mo")
    if history_daily.empty:
        return None, None, None, None
    return stock, info, history_daily, history_monthly

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
            stock, info, history_daily, history_monthly = fetch_financial_data(ticker_to_analyze)
            if history_daily is None:
                st.error(f"❌ Could not load data for {ticker_to_analyze}. Check if ticker is correct for Region: {selected_country}.")
                st.stop()
            
            current_price = history_daily['Close'].iloc[-1]
            
            # 2. Run Full Analysis (EXACT SCANNER LOGIC)
            scan_results = analyze_single_ticker(ticker_to_analyze, history_daily, history_monthly)

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

            # --- DISPLAY RESULTS ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"{current_price:.2f} {info.get('currency', '')}")
            m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            m3.metric("🐂 Bullish Signals", scan_results['bullish_count'])
            m4.metric("🐻 Bearish Signals", scan_results['bearish_count'])
            
            t1, t2, t3, t4 = st.tabs(["🎯 FVG Scanner", "🏛️ Fundamentals", "🔭 Technicals", "💬 Social"])
            
            with t1:
                st.markdown("### 🎯 Multi-Timeframe FVG Analysis")
                st.caption("Using exact logic from Prath's Market Scanner")
                
                # --- BULLISH SIGNALS TABLE ---
                st.markdown("#### 🐂 Bullish Signals")
                
                bull_data = []
                for tf in ['1D', '1W', '1M']:
                    row = {
                        'Timeframe': tf,
                        'Order Block': '✅' if scan_results.get(f'Bull_OB_{tf}') else '❌',
                        'FVG + Swing Break': '✅' if scan_results.get(f'Bull_FVG_{tf}') else '❌',
                        'Reversal Cand': '✅' if scan_results.get(f'Bull_RevCand_{tf}') else '❌',
                        'iFVG': '✅' if scan_results.get(f'Bull_iFVG_{tf}') else '❌',
                    }
                    bull_data.append(row)
                
                bull_df = pd.DataFrame(bull_data)
                st.dataframe(bull_df, hide_index=True, use_container_width=True)
                
                # --- SUPPORT ZONES (HTF Confluence) ---
                st.markdown("#### 🏔️ Support Zones (Higher Timeframe Confluence)")
                
                support_found = []
                for tf in ['3M', '6M', '12M']:
                    if scan_results.get(f'Support_{tf}'):
                        support_found.append(tf)
                
                if len(support_found) >= 2:
                    st.success(f"✅ **CONFLUENCE DETECTED** - Price near unmitigated FVG zones on: {', '.join(support_found)}")
                    if scan_results['support_zones']:
                        with st.expander("View Zone Details", expanded=True):
                            for zone in scan_results['support_zones']:
                                st.json(zone)
                elif len(support_found) == 1:
                    st.warning(f"⚠️ Single timeframe support on {support_found[0]} - Not confluence (need 2+ TFs)")
                else:
                    st.info("❌ No higher timeframe support zones detected")
                
                # --- SQUEEZE ---
                st.markdown("#### 🔄 Squeeze Status (Choppiness > 59)")
                squeeze_col1, squeeze_col2 = st.columns(2)
                with squeeze_col1:
                    if scan_results.get('Squeeze_1D'):
                        st.success("✅ Daily Squeeze Active")
                    else:
                        st.info("❌ No Daily Squeeze")
                with squeeze_col2:
                    if scan_results.get('Squeeze_1W'):
                        st.success("✅ Weekly Squeeze Active")
                    else:
                        st.info("❌ No Weekly Squeeze")
                
                st.divider()
                
                # --- BEARISH SIGNALS TABLE ---
                st.markdown("#### 🐻 Bearish Signals")
                
                bear_data = []
                for tf in ['1D', '1W', '1M']:
                    row = {
                        'Timeframe': tf,
                        'Order Block': '✅' if scan_results.get(f'Bear_OB_{tf}') else '❌',
                        'FVG + Swing Break': '✅' if scan_results.get(f'Bear_FVG_{tf}') else '❌',
                        'Reversal Cand': '✅' if scan_results.get(f'Bear_RevCand_{tf}') else '❌',
                        'iFVG': '✅' if scan_results.get(f'Bear_iFVG_{tf}') else '❌',
                    }
                    bear_data.append(row)
                
                bear_df = pd.DataFrame(bear_data)
                st.dataframe(bear_df, hide_index=True, use_container_width=True)
                
                # --- EXHAUSTION ---
                if scan_results.get('Exhaustion'):
                    st.error("⚠️ **EXHAUSTION DETECTED** - Weekly Choppiness < 25 (Trend may reverse)")
            
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