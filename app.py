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
    .fvg-card.bullish { border-left-color: #2ea043; } /* Green border for Bullish */
    .fvg-card.bearish { border-left-color: #da3633; } /* Red border for Bearish */
    
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
    Loads US, India, and Canada stock lists from CSVs.
    Standardizes columns and adds Yahoo Finance suffixes.
    """
    master_df = pd.DataFrame()

    def load_file(filename, country, suffix=""):
        try:
            # Try reading with default comma separator first
            df = pd.read_csv(filename)
            # If only 1 column is found, it might be pipe-separated
            if len(df.columns) < 2:
                df = pd.read_csv(filename, sep='|')
            
            df.columns = [c.strip() for c in df.columns]
            
            # Smart column detection
            cols = [c.lower() for c in df.columns]
            sym_col = df.columns[cols.index('ticker')] if 'ticker' in cols else (df.columns[cols.index('symbol')] if 'symbol' in cols else df.columns[0])
            name_col = df.columns[cols.index('company name')] if 'company name' in cols else df.columns[1]

            # Clean Ticker
            df['Yahoo_Ticker'] = df[sym_col].astype(str).str.replace(suffix, '', regex=False).str.strip() + suffix
            df['Display_Name'] = df[name_col].astype(str).str.strip()
            df['Country'] = country
            
            return df[['Yahoo_Ticker', 'Display_Name', 'Country']]
        except Exception:
            return pd.DataFrame()

    # Load India
    india = load_file("india_stocks.csv", "India", ".NS")
    if not india.empty: master_df = pd.concat([master_df, india])

    # Load Canada
    canada = load_file("canada_stocks.csv", "Canada", ".TO")
    if not canada.empty: master_df = pd.concat([master_df, canada])

    # Load USA
    usa = load_file("us_stocks.csv", "USA", "")
    if not usa.empty: master_df = pd.concat([master_df, usa])
    
    return master_df

def smart_search(query, df):
    """
    Performs fuzzy search on Ticker and Company Name.
    """
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
# DATA FETCHING & RETURN CALCULATION
# ==========================================

def get_gemini_response(prompt):
    if model is None: return "**Error:** Model not initialized."
    try: return model.generate_content(prompt).text
    except Exception as e: return f"**Error:** {e}"

def fetch_financial_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    try: info = stock.info
    except: info = {}
    
    # Attempt to fetch balance sheet for Debt-to-Asset Calc
    try:
        bs = stock.balance_sheet
        if not bs.empty:
            info['totalAssets'] = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else None
    except:
        pass

    history_1h = stock.history(period="730d", interval="1h")
    history_daily = stock.history(period="max", interval="1d")
    history_monthly = stock.history(period="max", interval="1mo")
    
    if history_daily.empty: return None, None, None, None, None, None
    
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
    all_data = []
    try:
        time.sleep(0.5)
        with DDGS() as ddgs:
            try:
                news = list(ddgs.news(f"{ticker} stock {company_name}", timelimit="w", max_results=5))
                for n in news: all_data.append(f"[News] {n.get('title')}")
            except: pass
    except: pass
    return "\n".join(all_data[:5]) if all_data else "No recent news."

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
            st.warning("No matches found. Using exact input.")
            ticker_to_analyze = user_query.upper()
    else:
        ticker_to_analyze = user_query.upper()

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
            
            # Calculate metrics locally if possible to help prompt
            total_debt = info.get('totalDebt', 0)
            total_assets = info.get('totalAssets', 1) # avoid div/0
            debt_to_assets = total_debt / total_assets if total_assets else "N/A"
            
            # 2. FUNDAMENTAL PROMPT (REVAMPED)
            fund_prompt = f"""
            You are an elite Equity Research Analyst. Perform a rigorous fundamental analysis of {company_name} ({ticker_to_analyze}) using the EXACT data and grading rubric below.

            ### 📊 RAW FINANCIAL DATA
            * **Current Price:** {current_price}
            * **Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}
            * **P/E Ratio:** {info.get('trailingPE', 'N/A')}
            * **PEG Ratio:** {info.get('pegRatio', 'N/A')}
            * **EV/EBITDA:** {info.get('enterpriseToEbitda', 'N/A')}
            * **Return on Equity (ROE):** {info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 'N/A'}%
            * **Return on Assets (Proxy for ROI):** {info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 'N/A'}%
            * **Profit Margin (Net Margin):** {info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 'N/A'}%
            * **Debt-to-Equity:** {info.get('debtToEquity', 'N/A')}
            * **Total Debt:** {info.get('totalDebt', 'N/A')}
            * **Total Assets:** {info.get('totalAssets', 'N/A')}
            * **Debt-to-Asset Ratio:** {debt_to_assets}
            * **Free Cash Flow:** {info.get('freeCashflow', 'N/A')}
            * **Revenue Growth:** {info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 'N/A'}%
            * **Earnings Growth:** {info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 'N/A'}%
            * **Dividend Yield:** {info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 'N/A'}%
            * **1-Year Return:** {returns['1Y']}%
            * **3-Year Return:** {returns['3Y']}%
            * **5-Year Return:** {returns['5Y']}%

            ### 📝 GRADING RUBRIC (STRICTLY APPLY THIS)
            Compare metrics against these thresholds to assign a GRADE (Blue, Green, Orange, Red).
            **IMPORTANT:** Also compare these metrics against typical {info.get('sector', 'N/A')} sector benchmarks in your analysis.

            **1. VALUATION & PROFITABILITY:**
            * **P/E Ratio:** 0-16 (🔵 Great/Undervalued), 16-25 (🟢 Good/Fair), 25-35 (🟠 Caution/Expensive), >35 or <0 (🔴 Bad/Overvalued)
            * **ROE:** >15% (🔵 Great/High Efficiency), 5-15% (🟢 Good), <5% (🔴 Bad/Inefficient)
            * **Net Margin:** >15% (🔵 Great/High Profit), 5-15% (🟢 Good/Healthy), 0-5% (🟠 Caution/Low), <0% (🔴 Bad/Loss)
            * **ROI:** >10% (🔵 Great), 5-10% (🟢 Good), <5% (🔴 Bad)

            **2. DEBT & HEALTH:**
            * **Debt-to-Equity:** ≤0.6 (🔵 Great/Conservative), 0.6-1.0 (🟢 Good/Manageable), 1.0-1.5 (🟠 Caution/Leveraged), ≥1.5 (🔴 Bad/High Risk)
            * **Debt-to-Asset:** <0.3 (🔵 Great/Asset Rich), 0.3-0.5 (🟢 Good), 0.5-0.75 (🟠 Caution), >0.75 (🔴 Bad/Debt Heavy)

            **3. MOMENTUM / PERFORMANCE:**
            * **1-Year Return:** >75% (🟠 Hot/Very Hot), 25-75% (🟢 Strong Trend), 15-25% (⚫ Neutral), <15% (🔴 Weak)
            * **3-Year Return:** >200% (🟠 Multi-bagger), 45-150% (🟢 Consistent), 30-45% (⚫ Neutral), <30% (🔴 Stagnant)
            * **5-Year Return:** >300% (🟠 Outlier), 100-200% (🟢 Doubler), 75-100% (⚫ Neutral), <75% (🔴 Underperformer)

            ### 📢 OUTPUT REQUIREMENTS
            1.  **Metric Analysis Table:** Create a Markdown table with columns: **Metric | Value | Grade (Color) | Verdict**. Use emojis (🔵, 🟢, 🟠, 🔴, ⚫) for the grades.
            2.  **Profitability Deep Dive:** Analyze FCF, ROIC, and ROE.
            3.  **Valuation Context:** Analyze P/E, PEG, and EV/EBITDA against SECTOR PEERS.
            4.  **Growth & Moat:** Discuss Revenue/EPS growth and Competitive Advantage (Moat).
            5.  **Financial Health:** Analyze Net Debt, Debt-to-Equity, and Debt-to-Assets.
            6.  **Sector Comparison:** Explicitly state how these metrics compare to the average {info.get('sector', 'N/A')} company.
            """

            # 3. TECHNICAL & SENTIMENT PROMPTS
            tech_prompt = f"""
            Analyze Technicals for {ticker_to_analyze}.
            Price: {current_price}
            Returns: 1Y: {returns['1Y']}%, 3Y: {returns['3Y']}%.
            Apply: Wyckoff, Price Action, Moving Averages.
            Verdict: Bullish/Bearish/Neutral?
            """
            
            soc_prompt = f"""
            Analyze Sentiment for {ticker_to_analyze}.
            News Context: {get_social_sentiment_data(ticker_to_analyze, company_name)}
            Provide: Sentiment Score (1-10), Key Drivers, Risks.
            """

            dashboard_prompt = f"""
            Create a JSON summary for {ticker_to_analyze}.
            Price: {current_price}
            Fundamentals: Use the data provided in previous prompts.
            Technicals: Trend is based on {returns['1Y']}% 1Y return.
            
            Output strictly valid JSON:
            {{
                "fundamentals_verdict": "Bullish/Bearish/Neutral",
                "fundamentals_rating": "X/10",
                "technicals_verdict": "Bullish/Bearish/Neutral",
                "risk_level": "Low/Medium/High",
                "sentiment_verdict": "Positive/Negative",
                "key_driver": "Main stock driver",
                "one_year_return": "{returns['1Y']}%"
            }}
            """

            # 4. Generate & Display
            fund_an = get_gemini_response(fund_prompt)
            tech_an = get_gemini_response(tech_prompt)
            soc_an = get_gemini_response(soc_prompt)
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
                st.subheader("Multi-Timeframe FVG Confluence")
                
                # --- Helper to render cards ---
                def render_confluence_card(pair_name, data, mode="bullish"):
                    t1, t2 = pair_name.split(', ')
                    z1 = data['tf1']
                    z2 = data['tf2']
                    thresholds = data.get('thresholds', (0,0))
                    
                    css_class = "bullish" if mode == "bullish" else "bearish"
                    status_class_1 = "status-inside" if z1['status'] == "INSIDE" else "status-near"
                    status_class_2 = "status-inside" if z2['status'] == "INSIDE" else "status-near"
                    
                    st.markdown(f"""
                    <div class="fvg-card {css_class}">
                        <div class="fvg-header">{pair_name} Confluence</div>
                        <div style="display: flex; justify-content: space-between;">
                            <div style="flex: 1; padding-right: 10px;">
                                <div class="fvg-sublabel">{t1} ZONE ({thresholds[0]}% Tol)</div>
                                <div class="fvg-value">${z1['bottom']:.2f} - ${z1['top']:.2f}</div>
                                <span class="status-badge {status_class_1}">{z1['status']}</span>
                            </div>
                            <div style="border-right: 1px solid #444; margin: 0 15px;"></div>
                            <div style="flex: 1;">
                                <div class="fvg-sublabel">{t2} ZONE ({thresholds[1]}% Tol)</div>
                                <div class="fvg-value">${z2['bottom']:.2f} - ${z2['top']:.2f}</div>
                                <span class="status-badge {status_class_2}">{z2['status']}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Render Bullish
                if fvg_results and fvg_results['bullish_confluence']:
                    st.success("🐂 **Bullish Zones Detected**")
                    for pair, data in fvg_results['bullish_confluence'].items():
                        render_confluence_card(pair, data, "bullish")
                elif fvg_results:
                    st.info("No Bullish Confluence Zones")

                # Render Bearish
                if fvg_results and fvg_results['bearish_confluence']:
                    st.error("🐻 **Bearish Zones Detected**")
                    for pair, data in fvg_results['bearish_confluence'].items():
                        render_confluence_card(pair, data, "bearish")
                elif fvg_results:
                    st.info("No Bearish Confluence Zones")

            with t2: st.markdown(fund_an)
            
            with t3:
                fig = go.Figure(data=[go.Candlestick(x=h1d.index, open=h1d['Open'], high=h1d['High'], low=h1d['Low'], close=h1d['Close'])])
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(tech_an)
                
            with t4: st.markdown(soc_an)