import streamlit as st
import yfinance as yf
from yahooquery import Ticker as YQTicker  # FASTER FUNDAMENTALS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import time
import json
import difflib
from datetime import datetime, timedelta

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
    /* Global App Style */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #d4af37 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
    .stMetricValue { color: #d4af37 !important; }
    
    /* Buttons */
    .stButton>button { background-color: #1f6feb; color: white; border: none; border-radius: 4px; font-weight: bold; }
    .stButton>button:hover { background-color: #388bfd; }
    
    /* Inputs */
    .stTextInput>div>div>input { color: #ffffff; background-color: #161b22; }
    
    /* FVG Card Styling */
    .fvg-card {
        background-color: #161b22;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #444;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .fvg-card.bullish { border-left-color: #2ea043; }
    .fvg-card.bearish { border-left-color: #da3633; }
    
    .fvg-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #e0e0e0;
        margin-bottom: 10px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 5px;
    }
    
    .fvg-sublabel { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
    .fvg-value { font-size: 1rem; font-weight: 500; color: #ffffff; }
    
    .status-badge {
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .status-inside { background-color: rgba(46, 160, 67, 0.2); color: #2ea043; border: 1px solid #2ea043; }
    .status-near { background-color: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #d29922; }
    
    /* Table Styling */
    div[data-testid="stMarkdownContainer"] table { width: 100%; border-collapse: collapse; }
    div[data-testid="stMarkdownContainer"] th { background-color: #262730; color: #d4af37; padding: 10px; text-align: left; }
    div[data-testid="stMarkdownContainer"] td { border-bottom: 1px solid #444; padding: 8px; }
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
    Loads US, India, and Canada stock lists from CSVs/TXTs.
    Standardizes columns and adds Yahoo Finance suffixes.
    """
    master_df = pd.DataFrame()

    def load_file(filename, country, suffix=""):
        try:
            # Try reading with default comma separator first
            df = pd.read_csv(filename)
            # If only 1 column is found, it might be pipe-separated (like your txt files)
            if len(df.columns) < 2:
                df = pd.read_csv(filename, sep='|')
            
            df.columns = [c.strip() for c in df.columns]
            
            # Smart column detection
            cols = [c.lower() for c in df.columns]
            # Look for common column names
            sym_col = next((df.columns[i] for i, c in enumerate(cols) if 'ticker' in c or 'symbol' in c), df.columns[0])
            name_col = next((df.columns[i] for i, c in enumerate(cols) if 'company' in c or 'name' in c), df.columns[1])

            # Clean Ticker
            df['Yahoo_Ticker'] = df[sym_col].astype(str).str.replace(suffix, '', regex=False).str.strip() + suffix
            df['Display_Name'] = df[name_col].astype(str).str.strip()
            df['Country'] = country
            
            return df[['Yahoo_Ticker', 'Display_Name', 'Country']]
        except Exception:
            return pd.DataFrame()

    # Load India
    india = load_file("India Stocks List.txt", "India", ".NS")
    if not india.empty: master_df = pd.concat([master_df, india])

    # Load Canada
    canada = load_file("Canada Stocks List.txt", "Canada", ".TO")
    if not canada.empty: master_df = pd.concat([master_df, canada])

    # Load USA
    usa = load_file("US Stocks List.txt", "USA", "")
    if not usa.empty: master_df = pd.concat([master_df, usa])
    
    return master_df

def smart_search(query, df):
    """Performs fuzzy search on Ticker and Company Name."""
    if df.empty or not query: return []
    query = query.lower().strip()
    
    # 1. Exact/Substring Match
    mask = (
        df['Yahoo_Ticker'].str.lower().str.contains(query, na=False) | 
        df['Display_Name'].str.lower().str.contains(query, na=False)
    )
    direct_matches = df[mask].head(15).to_dict('records')
    
    # 2. Fuzzy Match
    if len(direct_matches) < 5:
        all_names = df['Display_Name'].tolist()
        close_names = difflib.get_close_matches(query, all_names, n=5, cutoff=0.5)
        fuzzy_rows = df[df['Display_Name'].isin(close_names)].to_dict('records')
        
        seen = set(d['Yahoo_Ticker'] for d in direct_matches)
        for item in fuzzy_rows:
            if item['Yahoo_Ticker'] not in seen:
                direct_matches.append(item)
                
    return direct_matches

# Load data once
stock_db = load_and_prep_data()

# ==========================================
# FVG ENGINE
# ==========================================

def calculate_bullish_fvgs(df):
    """Identifies ONLY Unmitigated Bullish FVGs (Support)."""
    if df is None or len(df) < 4: return []
    highs, lows = df['High'].values, df['Low'].values
    n = len(lows)
    min_future_low = np.full(n, np.inf)
    current_min = np.inf
    for i in range(n-2, -1, -1):
        current_min = min(current_min, lows[i])
        min_future_low[i] = current_min
    unmitigated_fvgs = []
    scan_limit = n - 3
    for i in range(scan_limit):
        if highs[i] < lows[i+2]:
            bottom, top = highs[i], lows[i+2]
            if i + 3 <= n - 2:
                if min_future_low[i+3] <= top: continue
            unmitigated_fvgs.append((bottom, top))
    return unmitigated_fvgs

def calculate_bearish_fvgs(df):
    """Identifies ONLY Unmitigated Bearish FVGs (Resistance)."""
    if df is None or len(df) < 4: return []
    highs, lows = df['High'].values, df['Low'].values
    n = len(highs)
    max_future_high = np.full(n, -np.inf)
    current_max = -np.inf
    for i in range(n-2, -1, -1):
        current_max = max(current_max, highs[i])
        max_future_high[i] = current_max
    unmitigated_fvgs = []
    scan_limit = n - 3
    for i in range(scan_limit):
        if lows[i] > highs[i+2]:
            top, bottom = lows[i], highs[i+2]
            if i + 3 <= n - 2:
                if max_future_high[i+3] >= bottom: continue
            unmitigated_fvgs.append((bottom, top))
    return unmitigated_fvgs

def is_near_fvg(price, fvgs, threshold_pct):
    if not fvgs: return False, None
    threshold = threshold_pct / 100.0
    for bot, top in fvgs:
        if bot <= price <= top: return True, {'bottom': bot, 'top': top, 'status': 'INSIDE'}
        min_dist = min(abs(price - top), abs(price - bot))
        if (min_dist / price) <= threshold: return True, {'bottom': bot, 'top': top, 'status': 'NEAR', 'distance_pct': (min_dist / price) * 100}
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
    if timeframe == '3M': df['Group'] = (df['Month'] - 1) // 3; grouper = ['Year', 'Group']
    elif timeframe == '6M': df['Group'] = (df['Month'] - 1) // 6; grouper = ['Year', 'Group']
    elif timeframe == '12M': df['Group'] = 0; grouper = ['Year']
    else: return df 

    resampled = df.groupby(grouper).agg(agg_dict).dropna()
    resampled = resampled.sort_index()
    return resampled

def analyze_fvg_confluence(ticker, df_1h, df_1d, df_1mo):
    if df_1d is None or df_1d.empty: return None
    current_price = df_1d['Close'].iloc[-1]
    
    data_frames = {'1D': df_1d, '1W': resample_custom(df_1d, '1W')}
    if df_1h is not None and not df_1h.empty:
        data_frames['1H'] = df_1h
        data_frames['4H'] = resample_custom(df_1h, '4H')
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
    
    results = {'current_price': current_price, 'bullish_confluence': {}, 'bearish_confluence': {}}
    for tf1, tf2, t1, t2 in confluence_pairs:
        if tf1 in bullish_fvgs and tf2 in bullish_fvgs:
            n1, z1 = is_near_fvg(current_price, bullish_fvgs[tf1], t1)
            n2, z2 = is_near_fvg(current_price, bullish_fvgs[tf2], t2)
            if n1 and n2: results['bullish_confluence'][f"{tf1}, {tf2}"] = {'tf1': z1, 'tf2': z2, 'thresholds': (t1, t2)}
        
        if tf1 in bearish_fvgs and tf2 in bearish_fvgs:
            n1, z1 = is_near_fvg(current_price, bearish_fvgs[tf1], t1)
            n2, z2 = is_near_fvg(current_price, bearish_fvgs[tf2], t2)
            if n1 and n2: results['bearish_confluence'][f"{tf1}, {tf2}"] = {'tf1': z1, 'tf2': z2, 'thresholds': (t1, t2)}
    
    return results

# ==========================================
# DATA FETCHING (FAST FUNDAMENTALS)
# ==========================================

def get_gemini_response(prompt):
    if model is None: return "**Error:** Model not initialized."
    try: return model.generate_content(prompt).text
    except Exception as e: return f"**Error:** {e}"

def fetch_financial_data(ticker_symbol):
    """
    Hybrid Fetcher:
    - History from yfinance (Fast for charts)
    - Fundamentals from yahooquery (Fast for data)
    """
    # 1. Fetch History (yfinance is good for this)
    stock = yf.Ticker(ticker_symbol)
    history_1h = stock.history(period="730d", interval="1h")
    history_daily = stock.history(period="max", interval="1d")
    history_monthly = stock.history(period="max", interval="1mo")
    
    if history_daily.empty: return None, None, None, None, None, None
    
    # 2. Fetch Fundamentals (yahooquery is faster/reliable)
    info = {}
    try:
        yq = YQTicker(ticker_symbol)
        # Fetch all relevant modules in ONE request
        modules = 'summaryDetail assetProfile financialData defaultKeyStatistics earnings'
        yq_data = yq.get_modules(modules)
        
        # Flatten the nested JSON structure
        if isinstance(yq_data, dict) and ticker_symbol in yq_data:
            data = yq_data[ticker_symbol]
            if isinstance(data, dict):
                # Helper to safely get nested keys
                def get_val(module, key):
                    return data.get(module, {}).get(key, None)

                info = {
                    'currency': get_val('financialData', 'financialCurrency'),
                    'sector': get_val('assetProfile', 'sector'),
                    'industry': get_val('assetProfile', 'industry'),
                    'longName': get_val('quoteType', 'longName'), # Might need quoteType module
                    'trailingPE': get_val('summaryDetail', 'trailingPE'),
                    'pegRatio': get_val('defaultKeyStatistics', 'pegRatio'),
                    'priceToBook': get_val('defaultKeyStatistics', 'priceToBook'),
                    'enterpriseToEbitda': get_val('defaultKeyStatistics', 'enterpriseToEbitda'),
                    'returnOnEquity': get_val('financialData', 'returnOnEquity'),
                    'returnOnAssets': get_val('financialData', 'returnOnAssets'),
                    'profitMargins': get_val('financialData', 'profitMargins'),
                    'debtToEquity': get_val('financialData', 'debtToEquity'),
                    'totalDebt': get_val('financialData', 'totalDebt'),
                    'totalCash': get_val('financialData', 'totalCash'),
                    'freeCashflow': get_val('financialData', 'freeCashflow'),
                    'revenueGrowth': get_val('financialData', 'revenueGrowth'),
                    'earningsGrowth': get_val('financialData', 'earningsGrowth'),
                    'dividendYield': get_val('summaryDetail', 'dividendYield'),
                }
                
                # Fetch Balance Sheet separately for Total Assets if needed
                try:
                    bs = yq.balance_sheet()
                    if not bs.empty and 'TotalAssets' in bs.columns:
                        info['totalAssets'] = bs['TotalAssets'].iloc[-1]
                except: pass
                
    except Exception as e:
        print(f"YahooQuery Failed: {e}")
        # Fallback to empty info, chart will still load
        pass

    # Calculate Returns for Grading
    current_price = history_daily['Close'].iloc[-1]
    returns = {"1Y": "N/A", "3Y": "N/A", "5Y": "N/A"}
    
    def get_pct_change(days_ago):
        target_date = history_daily.index[-1] - timedelta(days=days_ago)
        try:
            idx = history_daily.index.get_indexer([target_date], method='nearest')[0]
            old_price = history_daily['Close'].iloc[idx]
            return ((current_price - old_price) / old_price) * 100
        except:
            return "N/A"

    if len(history_daily) > 252: returns["1Y"] = get_pct_change(365)
    if len(history_daily) > 756: returns["3Y"] = get_pct_change(365*3)
    if len(history_daily) > 1260: returns["5Y"] = get_pct_change(365*5)
    
    return stock, info, history_1h, history_daily, history_monthly, returns

def get_social_sentiment_data(ticker, company_name=""):
    # Simulated simple data as DDGS can be slow/rate-limited
    return f"Fetching real-time social data for {ticker}..."

# ==========================================
# UI WORKFLOW
# ==========================================
st.markdown("### 1️⃣ Identify Target")

col_search, col_btn = st.columns([3, 1])
ticker_to_analyze = None

with col_search:
    user_query = st.text_input("Search Company or Ticker", placeholder="e.g. Reliance, Apple, Tesla (Typos allowed)")

if user_query:
    if stock_db is not None and not stock_db.empty:
        results = smart_search(user_query, stock_db)
        if results:
            options = [f"{r['Yahoo_Ticker']} | {r['Display_Name']} [{r['Country']}]" for r in results]
            selection = st.selectbox("Select Correct Stock:", options)
            if selection: ticker_to_analyze = selection.split(" | ")[0]
        else:
            st.warning("No matches found in database. Using manual input fallback.")
            ticker_to_analyze = user_query.upper().strip()
    else:
        # Fallback if DB load fails
        ticker_to_analyze = user_query.upper().strip()

# --- ANALYSIS EXECUTION ---
if ticker_to_analyze:
    st.divider()
    st.markdown(f"### 2️⃣ Analyze: **{ticker_to_analyze}**")
    
    if st.button("🚀 Launch Strategic Deep Dive", use_container_width=True):
        if not api_key:
            st.error("⚠️ API Key Missing.")
            st.stop()
        
        with st.spinner(f"Running Analysis on {ticker_to_analyze}..."):
            
            # 1. Fetch Data
            stock, info, h1h, h1d, h1m, returns = fetch_financial_data(ticker_to_analyze)
            if h1d is None:
                st.error("❌ Data fetch failed.")
                st.stop()
            
            current_price = h1d['Close'].iloc[-1]
            fvg_results = analyze_fvg_confluence(ticker_to_analyze, h1h, h1d, h1m)
            company_name = info.get('longName', ticker_to_analyze)
            
            # Derived Metrics for Prompt
            total_debt = info.get('totalDebt', 0)
            total_assets = info.get('totalAssets', 1) 
            debt_to_assets = total_debt / total_assets if total_assets and total_debt else "N/A"
            
            # 2. FUNDAMENTAL PROMPT
            fund_prompt = f"""
            You are an elite Equity Research Analyst. Perform a rigorous fundamental analysis of {company_name} ({ticker_to_analyze}) using the EXACT data below.

            ### 📊 RAW FINANCIAL DATA
            * **Price:** {current_price} | **Sector:** {info.get('sector', 'N/A')}
            * **P/E:** {info.get('trailingPE', 'N/A')} | **PEG:** {info.get('pegRatio', 'N/A')}
            * **EV/EBITDA:** {info.get('enterpriseToEbitda', 'N/A')}
            * **ROE:** {info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 'N/A'}%
            * **Net Margin:** {info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 'N/A'}%
            * **Debt-to-Equity:** {info.get('debtToEquity', 'N/A')} | **Debt-to-Assets:** {debt_to_assets}
            * **Revenue Growth:** {info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 'N/A'}%
            * **Returns:** 1Y: {returns['1Y']}% | 3Y: {returns['3Y']}% | 5Y: {returns['5Y']}%

            ### 📝 GRADING RUBRIC (STRICTLY APPLY THIS)
            Compare metrics against these thresholds to assign a GRADE (Blue, Green, Orange, Red).

            **1. VALUATION & PROFITABILITY:**
            * **P/E Ratio:** 0-16 (🔵 Great), 16-25 (🟢 Good), 25-35 (🟠 Caution), >35 (🔴 Bad)
            * **ROE:** >15% (🔵 Great), 5-15% (🟢 Good), <5% (🔴 Bad)
            * **Net Margin:** >15% (🔵 Great), 5-15% (🟢 Good), 0-5% (🟠 Caution), <0% (🔴 Bad)

            **2. DEBT & HEALTH:**
            * **Debt-to-Equity:** ≤0.6 (🔵 Great), 0.6-1.0 (🟢 Good), 1.0-1.5 (🟠 Caution), ≥1.5 (🔴 Bad)
            * **Debt-to-Asset:** <0.3 (🔵 Great), 0.3-0.5 (🟢 Good), >0.75 (🔴 Bad)

            **3. MOMENTUM:**
            * **1-Year Return:** >75% (🟠 Hot), 25-75% (🟢 Strong), <15% (🔴 Weak)

            ### 📢 OUTPUT REQUIREMENTS
            1.  **Metric Table:** Markdown table: Metric | Value | Grade (Color) | Verdict.
            2.  **Profitability:** Analyze FCF, ROIC, ROE.
            3.  **Valuation:** Analyze P/E, PEG vs Peers.
            4.  **Health:** Analyze Debt load.
            """

            # 3. TECHNICAL & DASHBOARD PROMPTS
            tech_prompt = f"Analyze Technicals for {ticker_to_analyze}. Price: {current_price}. Returns 1Y: {returns['1Y']}%. Trend?"
            dashboard_prompt = f"""
            Create JSON summary for {ticker_to_analyze}. Price: {current_price}.
            Fund Verdict: Based on P/E {info.get('trailingPE','N/A')} and ROE.
            Tech Verdict: Based on 1Y Return {returns['1Y']}%.
            Output JSON: {{ "fundamentals_verdict": "string", "fundamentals_rating": "X/10", "technicals_verdict": "string", "risk_level": "Low/Med/High", "sentiment_verdict": "string", "key_driver": "string" }}
            """

            # 4. Generate & Display
            fund_an = get_gemini_response(fund_prompt)
            tech_an = get_gemini_response(tech_prompt)
            dash_raw = get_gemini_response(dashboard_prompt)

            try:
                dash_data = json.loads(dash_raw.strip().replace("```json", "").replace("```", ""))
            except:
                dash_data = {}

            # --- DISPLAY ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"{current_price:.2f} {info.get('currency', '')}")
            m2.metric("1-Year Return", f"{returns['1Y'] if isinstance(returns['1Y'], str) else round(returns['1Y'], 2)}%")
            m3.metric("🐂 Bull Confluences", len(fvg_results['bullish_confluence']) if fvg_results else 0)
            m4.metric("🐻 Bear Confluences", len(fvg_results['bearish_confluence']) if fvg_results else 0)

            t0, t1, t2, t3, t4 = st.tabs(["📊 Dashboard", "🎯 FVG Scanner", "🏛️ Fundamentals", "🔭 Technicals", "💬 Social"])

            with t0:
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"**Fundamentals:** {dash_data.get('fundamentals_verdict', 'N/A')} ({dash_data.get('fundamentals_rating', 'N/A')})")
                    st.success(f"**Technicals:** {dash_data.get('technicals_verdict', 'N/A')}")
                with c2:
                    st.warning(f"**Sentiment:** {dash_data.get('sentiment_verdict', 'N/A')}")
                    st.error(f"**Risk Level:** {dash_data.get('risk_level', 'N/A')}")
                st.write(f"**Key Driver:** {dash_data.get('key_driver', 'N/A')}")

            with t1:
                # Reuse the nice Card UI from previous turn
                def render_card(pair, data, mode):
                    css = "bullish" if mode=="bullish" else "bearish"
                    st.markdown(f"""<div class="fvg-card {css}">
                        <div class="fvg-header">{pair}</div>
                        <div>{data['tf1_zone']['status']} | {data['tf2_zone']['status']}</div>
                    </div>""", unsafe_allow_html=True)
                
                if fvg_results['bullish_confluence']:
                    st.success("🐂 **Bullish Zones**")
                    for p, d in fvg_results['bullish_confluence'].items(): render_card(p, d, "bullish")
                else: st.info("No Bullish Zones")
                
                if fvg_results['bearish_confluence']:
                    st.error("🐻 **Bearish Zones**")
                    for p, d in fvg_results['bearish_confluence'].items(): render_card(p, d, "bearish")
                else: st.info("No Bearish Zones")

            with t2: st.markdown(fund_an)
            
            with t3:
                fig = go.Figure(data=[go.Candlestick(x=h1d.index, open=h1d['Open'], high=h1d['High'], low=h1d['Low'], close=h1d['Close'])])
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(tech_an)
                
            with t4: st.write("Social sentiment analysis pending integration...")