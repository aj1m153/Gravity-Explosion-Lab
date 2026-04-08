# ─────────────────────────────────────────────────────────────────────────────
#  GRAVITY EXPLOSION LAB  ⚡
#  Volatility Explosion Tracker × Crypto Gravity Lines
#  Creator: tubakhxn  |  Combined by: Claude
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from io import StringIO

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gravity Explosion Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS / Theme ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #050a12;
    color: #c8d8e8;
}
.stApp { background-color: #050a12; }
.stSpinner > div { border-top-color: #00d4ff !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070e1a 0%, #050a12 100%);
    border-right: 1px solid #0f2a40;
}
section[data-testid="stSidebar"] * { color: #c8d8e8 !important; }

/* Slider */
.stSlider > div > div > div { background: #00d4ff !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0a1e30, #0f2a40);
    border: 1px solid #00d4ff44;
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 2px;
    font-size: 13px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0f2a40, #0a3050);
    border-color: #00d4ff;
    box-shadow: 0 0 12px rgba(0,212,255,0.25);
}

/* selectbox */
.stSelectbox > div > div {
    background: #070e1a;
    border-color: #0f2a40 !important;
    color: #c8d8e8;
    font-family: 'Share Tech Mono', monospace;
}

.metric-card {
    background: linear-gradient(135deg, #070e1a, #0a1628);
    border: 1px solid #0f3050;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.mc-green::before  { background: linear-gradient(90deg, #00ff88, transparent); }
.mc-red::before    { background: linear-gradient(90deg, #ff3366, transparent); }
.mc-yellow::before { background: linear-gradient(90deg, #ffcc00, transparent); }
.mc-cyan::before   { background: linear-gradient(90deg, #00d4ff, transparent); }
.mc-orange::before { background: linear-gradient(90deg, #ff8c00, transparent); }

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2a4a60;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 26px;
    font-weight: bold;
    line-height: 1;
}
.metric-sub {
    font-size: 10px;
    color: #456070;
    margin-top: 3px;
    font-family: 'Share Tech Mono', monospace;
}
.col-green  { color: #00ff88; }
.col-red    { color: #ff3366; }
.col-yellow { color: #ffcc00; }
.col-cyan   { color: #00d4ff; }
.col-orange { color: #ff8c00; }

.setup-score-wrap {
    text-align: center;
    padding: 22px 16px 18px;
    background: linear-gradient(135deg, #070e1a, #0a1628);
    border: 1px solid #0f3050;
    border-radius: 10px;
    margin-bottom: 14px;
}
.setup-score-num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 62px;
    font-weight: bold;
    line-height: 1;
}
.setup-score-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    color: #2a4a60;
    margin-top: 6px;
    text-transform: uppercase;
}
.setup-score-desc {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    margin-top: 8px;
}

.section-hdr {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    color: #2a4a60;
    text-transform: uppercase;
    border-bottom: 1px solid #0f2a40;
    padding-bottom: 5px;
    margin: 18px 0 10px 0;
}

.main-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 20px;
    color: #00d4ff;
    letter-spacing: 5px;
    text-transform: uppercase;
}
.main-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #1e3a50;
    letter-spacing: 3px;
    margin-top: 2px;
}

.signal-pill {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: bold;
    margin-bottom: 10px;
}
.pill-compression { background: rgba(0,212,255,0.12); color: #00d4ff; border: 1px solid rgba(0,212,255,0.35); }
.pill-explosion   { background: rgba(255,140,0,0.12); color: #ff8c00; border: 1px solid rgba(255,140,0,0.35); }
.pill-neutral     { background: rgba(42,74,96,0.20);  color: #456070; border: 1px solid rgba(42,74,96,0.40); }
.pill-aligned     { background: rgba(0,255,136,0.12); color: #00ff88; border: 1px solid rgba(0,255,136,0.35); }

.level-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 10px;
    margin-bottom: 3px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 30:
            raise ValueError("Insufficient data from yfinance")
        # Flatten MultiIndex columns if present
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception:
        return _mock_data()


def _mock_data() -> pd.DataFrame:
    np.random.seed(99)
    n = 350
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='1h')
    p = [45000.0]
    for _ in range(n - 1):
        shock = np.random.choice([1, 5], p=[0.95, 0.05])
        p.append(p[-1] * (1 + np.random.normal(0, 0.006 * shock)))
    p = np.array(p)
    noise = lambda s: np.random.uniform(-s, s, n)
    return pd.DataFrame({
        'Open':   p * (1 + noise(0.002)),
        'High':   p * (1 + abs(noise(0.010))),
        'Low':    p * (1 - abs(noise(0.010))),
        'Close':  p,
        'Volume': np.random.randint(300, 6000, n).astype(float),
    }, index=dates)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAVITY LINES  (from Crypto-Gravity-Lines)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gravity_lines(df: pd.DataFrame, n_levels: int = 6,
                           lookback: int = 120) -> np.ndarray:
    """
    Detect key support/resistance levels using KMeans clustering over:
      - Swing highs & lows (price structure)
      - Rolling closes (density / value area)
    Returns sorted array of price levels.
    """
    recent = df.tail(lookback)
    h = recent['High'].values
    l = recent['Low'].values
    c = recent['Close'].values

    pivots = list(c)  # closes as base density
    for i in range(2, len(recent) - 2):
        if h[i] > h[i-1] and h[i] > h[i+1] and h[i] > h[i-2] and h[i] > h[i+2]:
            pivots.append(h[i])
            pivots.append(h[i])  # double-weight swing highs
        if l[i] < l[i-1] and l[i] < l[i+1] and l[i] < l[i-2] and l[i] < l[i+2]:
            pivots.append(l[i])
            pivots.append(l[i])

    n_levels = min(n_levels, len(set(pivots)) - 1)
    if n_levels < 2:
        return np.linspace(df['Low'].min(), df['High'].max(), 6)

    pts = np.array(pivots).reshape(-1, 1)
    km = KMeans(n_clusters=n_levels, n_init=15, random_state=42)
    km.fit(pts)
    return np.sort(km.cluster_centers_.flatten())


def nearest_levels(price: float, levels: np.ndarray) -> dict:
    """Returns the nearest support and resistance around current price."""
    above = [(l, l - price) for l in levels if l > price]
    below = [(l, price - l) for l in levels if l <= price]
    result = {}
    if above:
        r = min(above, key=lambda x: x[1])
        result['resistance'] = {'level': r[0], 'dist': r[1], 'pct': r[1] / price}
    else:
        result['resistance'] = {'level': np.nan, 'dist': np.nan, 'pct': np.nan}
    if below:
        s = min(below, key=lambda x: x[1])
        result['support'] = {'level': s[0], 'dist': s[1], 'pct': s[1] / price}
    else:
        result['support'] = {'level': np.nan, 'dist': np.nan, 'pct': np.nan}

    dists = [r['pct'] for r in result.values() if not np.isnan(r['pct'])]
    result['nearest_pct'] = min(dists) if dists else 0.05
    return result


def level_touch_count(df: pd.DataFrame, levels: np.ndarray,
                       tolerance_pct: float = 0.004) -> dict:
    """Count how many times recent bars touched each level."""
    touches = {}
    c = df['Close'].values
    for lv in levels:
        tol = lv * tolerance_pct
        touches[lv] = int(np.sum(np.abs(c - lv) < tol))
    return touches


# ══════════════════════════════════════════════════════════════════════════════
#  VOLATILITY ENGINE  (from Volatility-Explosion-Tracker)
# ══════════════════════════════════════════════════════════════════════════════

def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling volatility analytics:
      - Annualized rolling std of returns
      - Vol percentile (historical rank)
      - Compression flag (bottom quartile)
      - Explosion flag (top quintile + rising)
      - ATR / ATR% for normalisation
    """
    d = df.copy()
    d['ret'] = d['Close'].pct_change()

    # Annualized vol (assumes hourly by default; scales per year's hours)
    annual_factor = np.sqrt(252 * 24)
    d['vol'] = d['ret'].rolling(window).std() * annual_factor

    # Historical percentile rank
    d['vol_pct'] = d['vol'].rolling(200, min_periods=window).rank(pct=True)

    # Compression: below 25th percentile on rolling basis
    d['is_compressed'] = d['vol_pct'] < 0.25
    # Explosion: above 80th percentile AND vol increasing
    d['is_explosion']  = (d['vol_pct'] > 0.80) & (d['vol'] > d['vol'].shift(1))

    # Vol z-score
    v_mean = d['vol'].rolling(150, min_periods=window).mean()
    v_std  = d['vol'].rolling(150, min_periods=window).std().replace(0, np.nan)
    d['vol_z'] = (d['vol'] - v_mean) / v_std

    # ATR
    d['atr'] = (d['High'] - d['Low']).rolling(14).mean()
    d['atr_pct'] = (d['atr'] / d['Close']).replace(0, np.nan)

    # Bollinger Band width as secondary compression metric
    mid   = d['Close'].rolling(window).mean()
    upper = mid + 2 * d['Close'].rolling(window).std()
    lower = mid - 2 * d['Close'].rolling(window).std()
    d['bb_width'] = (upper - lower) / mid

    return d


def get_compression_zones(df: pd.DataFrame, min_bars: int = 3) -> list[tuple]:
    """Return list of (start_ts, end_ts) for each compression run."""
    flags = df['is_compressed'].fillna(False)
    zones, in_zone, start, start_i = [], False, None, 0
    idx_list = list(df.index)
    for i, (ts, val) in enumerate(flags.items()):
        if val and not in_zone:
            in_zone, start, start_i = True, ts, i
        elif not val and in_zone:
            in_zone = False
            if (i - start_i) >= min_bars:
                zones.append((start, ts))
    # Capture an open compression zone at the end of the data
    if in_zone and (len(idx_list) - 1 - start_i) >= min_bars:
        zones.append((start, idx_list[-1]))
    return zones


# ══════════════════════════════════════════════════════════════════════════════
#  ML BREAKOUT PREDICTOR  (merged from both repos)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def train_model(_df_hash, df_json: str, levels_str: str):
    """
    Gradient Boosted Classifier predicting whether vol will surge >50% in 5 bars.
    Features blend signals from BOTH repos:
      - From Volatility-Explosion-Tracker: vol_pct, vol_z, vol_trend, atr_pct
      - From Crypto-Gravity-Lines:         dist_to_nearest, bb_width
    """
    df     = pd.read_json(StringIO(df_json))
    levels = np.array([float(x) for x in levels_str.split(',')])

    d = df.copy().dropna()
    d['dist_nearest'] = d['Close'].apply(
        lambda p: min(abs(p - lv) / p for lv in levels) if len(levels) else 0.02
    )
    d['vol_trend']  = d['vol'].diff(5)
    d['vol_accel']  = d['vol_trend'].diff(3)
    d['price_mom']  = d['Close'].pct_change(5)
    d['atr_trend']  = d['atr_pct'].diff(5)

    future_vol = d['vol'].shift(-5)
    d['label'] = ((future_vol / (d['vol'] + 1e-9)) > 1.5).astype(int)

    FEATS = ['vol_pct', 'vol_z', 'dist_nearest', 'vol_trend',
             'vol_accel', 'price_mom', 'atr_pct', 'atr_trend', 'bb_width']

    valid = d[FEATS + ['label']].dropna()
    if len(valid) < 60 or len(valid['label'].unique()) < 2:
        return None

    X = valid[FEATS].values
    y = valid['label'].values
    split = int(len(X) * 0.80)
    X_tr, y_tr = X[:split], y[:split]
    if len(np.unique(y_tr)) < 2:
        return None

    sc = StandardScaler().fit(X_tr)
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.08,
        subsample=0.8, random_state=42
    ).fit(sc.transform(X_tr), y_tr)

    return {'clf': clf, 'scaler': sc, 'features': FEATS}


def predict_prob(df: pd.DataFrame, levels: np.ndarray, bundle: dict | None) -> float:
    if bundle is None:
        return 0.5
    clf, sc, FEATS = bundle['clf'], bundle['scaler'], bundle['features']

    row = df.tail(1).copy()
    row['dist_nearest'] = row['Close'].apply(
        lambda p: min(abs(p - lv) / p for lv in levels) if len(levels) else 0.02
    )
    row['vol_trend'] = df['vol'].diff(5).iloc[-1]
    row['vol_accel'] = df['vol'].diff(5).diff(3).iloc[-1]
    row['price_mom'] = df['Close'].pct_change(5).iloc[-1]
    row['atr_trend'] = df['atr_pct'].diff(5).iloc[-1]

    try:
        x = row[FEATS].values
        if np.any(np.isnan(x)):
            return 0.5
        return float(clf.predict_proba(sc.transform(x))[0][1])
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  SETUP SCORE  (unique to this combined app)
# ══════════════════════════════════════════════════════════════════════════════

def setup_score(vol_pct: float, breakout_prob: float, dist_pct: float) -> tuple[int, str]:
    """
    0–100 composite score.
    High score = compressed volatility + price near gravity line + ML agrees.
    """
    # Max 40 pts: compression (lower vol_pct = better)
    comp = max(0.0, (0.30 - vol_pct) / 0.30) * 40
    # Max 30 pts: proximity to nearest gravity line (within 3%)
    prox = max(0.0, (0.03 - dist_pct) / 0.03) * 30
    # Max 30 pts: ML breakout probability
    ml   = breakout_prob * 30
    score = int(min(100, comp + prox + ml))

    if score >= 70:
        desc = "PRIME SETUP — compression + gravity alignment"
    elif score >= 50:
        desc = "DEVELOPING — watch for breakout trigger"
    elif score >= 30:
        desc = "EARLY STAGE — insufficient confluence"
    else:
        desc = "NO SETUP — vol expanded or far from levels"
    return score, desc


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

C = {  # Neon color palette
    'bg': '#050a12', 'panel': '#070e1a', 'grid': '#091828',
    'muted': '#2a4a60', 'text': '#c8d8e8',
    'green': '#00ff88', 'red': '#ff3366', 'cyan': '#00d4ff',
    'yellow': '#ffcc00', 'orange': '#ff8c00', 'purple': '#b060ff',
}

BASE_LAYOUT = dict(
    paper_bgcolor=C['bg'], plot_bgcolor=C['panel'],
    font=dict(family='Share Tech Mono', color=C['text'], size=10),
    margin=dict(l=10, r=65, t=36, b=20),
    hovermode='x unified',
    legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                font=dict(size=9, color=C['muted']), yanchor='bottom', y=1.01, x=0),
    xaxis=dict(gridcolor=C['grid'], zerolinecolor=C['grid'],
               showgrid=True, color=C['muted'], rangeslider=dict(visible=False)),
    yaxis=dict(gridcolor=C['grid'], zerolinecolor=C['grid'],
               showgrid=True, color=C['muted'], side='right'),
)


def main_chart(df: pd.DataFrame, levels: np.ndarray,
               zones: list, n_bars: int, touches: dict) -> go.Figure:
    plot = df.tail(n_bars)
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.70, 0.30],
        shared_xaxes=True, vertical_spacing=0.025,
    )

    # ── Compression zone shading ──────────────────────────────────────────────
    for z0, z1 in zones:
        if z1 < plot.index[0]:
            continue
        x0 = max(z0, plot.index[0])
        x1 = min(z1, plot.index[-1])
        for row in (1, 2):
            fig.add_vrect(x0=x0, x1=x1, row=row, col=1,
                          fillcolor='rgba(0,212,255,0.05)', line_width=0)

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=plot.index,
        open=plot['Open'], high=plot['High'],
        low=plot['Low'],   close=plot['Close'],
        increasing=dict(line=dict(color=C['green'], width=1),
                        fillcolor='rgba(0,255,136,0.33)'),
        decreasing=dict(line=dict(color=C['red'],   width=1),
                        fillcolor='rgba(255,51,102,0.33)'),
        name='Price', whiskerwidth=0.25,
    ), row=1, col=1)

    # ── Gravity Lines ─────────────────────────────────────────────────────────
    price_now = plot['Close'].iloc[-1]
    lo, hi    = plot['Low'].min(), plot['High'].max()
    span      = hi - lo or 1
    vis_lo, vis_hi = lo - span * 0.08, hi + span * 0.08

    max_touch = max(touches.values()) if touches else 1

    for lv in levels:
        if not (vis_lo <= lv <= vis_hi):
            continue
        is_res  = lv > price_now
        color   = C['red'] if is_res else C['green']
        dist_p  = abs(price_now - lv) / price_now
        prox    = max(0.0, 1.0 - dist_p / 0.04)           # glow when near
        touch_w = touches.get(lv, 0) / (max_touch or 1)   # weight by history
        width   = 0.8 + prox * 1.8 + touch_w * 0.8
        opacity = 0.25 + prox * 0.55 + touch_w * 0.15

        fig.add_hline(y=lv, row=1, col=1,
                      line=dict(color=color, width=width, dash='dot'),
                      opacity=min(1.0, opacity))
        tag = 'R' if is_res else 'S'
        fig.add_annotation(
            x=plot.index[-1], y=lv, xanchor='left', showarrow=False,
            text=f" {tag} {lv:,.0f}",
            font=dict(family='Share Tech Mono', size=9, color=color),
            row=1, col=1,
        )

    # ── Explosion markers on price chart ─────────────────────────────────────
    expl = plot[plot['is_explosion'] == True]
    if not expl.empty:
        fig.add_trace(go.Scatter(
            x=expl.index, y=expl['High'] * 1.006,
            mode='markers',
            marker=dict(symbol='triangle-up', size=11, color=C['orange'],
                        line=dict(color=C['orange'], width=1)),
            name='Explosion ▲', showlegend=True,
        ), row=1, col=1)

    # ── Vol area ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot['vol'],
        mode='lines', name='Volatility',
        line=dict(color=C['cyan'], width=1.5),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.07)',
    ), row=2, col=1)

    # Explosion markers on vol subplot
    if not expl.empty:
        fig.add_trace(go.Scatter(
            x=expl.index, y=expl['vol'],
            mode='markers', showlegend=False,
            marker=dict(symbol='circle', size=7, color=C['orange']),
        ), row=2, col=1)

    # Layout
    lo2 = BASE_LAYOUT.copy()
    lo2.update(dict(
        height=560, showlegend=True,
        title=dict(text='', font=dict(size=1)),
        xaxis2=dict(gridcolor=C['grid'], zerolinecolor=C['grid'],
                    showgrid=True, color=C['muted']),
        yaxis2=dict(gridcolor=C['grid'], zerolinecolor=C['grid'],
                    showgrid=True, color=C['muted'], side='right',
                    tickformat='.3f', title=dict(text='VOL', font=dict(size=9))),
    ))
    fig.update_layout(**lo2)
    return fig


def vol_gauge_chart(vol_pct: float, breakout_prob: float) -> go.Figure:
    fig = go.Figure()

    # Vol percentile gauge
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=round(vol_pct * 100, 1),
        number=dict(suffix='th', font=dict(family='Share Tech Mono', size=28,
                                            color=C['cyan'])),
        title=dict(text='VOL PERCENTILE',
                   font=dict(family='Share Tech Mono', size=9, color=C['muted'])),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=8, color=C['muted']),
                      tickcolor=C['muted'], dtick=25),
            bar=dict(color=C['cyan'], thickness=0.25),
            bgcolor=C['panel'],
            bordercolor=C['grid'], borderwidth=1,
            steps=[
                dict(range=[0, 25],  color='rgba(0,212,255,0.15)'),
                dict(range=[25, 60], color='rgba(42,74,96,0.10)'),
                dict(range=[60, 80], color='rgba(255,204,0,0.08)'),
                dict(range=[80, 100],color='rgba(255,51,102,0.15)'),
            ],
            threshold=dict(line=dict(color=C['orange'], width=2),
                           thickness=0.7, value=breakout_prob * 100),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))

    fig.update_layout(
        paper_bgcolor=C['bg'],
        plot_bgcolor=C['panel'],
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family='Share Tech Mono', color=C['text']),
    )
    return fig


def vol_histogram(df: pd.DataFrame) -> go.Figure:
    vol = df['vol'].dropna()
    cur = vol.iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vol, nbinsx=40,
        marker_color=C['cyan'], opacity=0.40, name='Vol History',
    ))
    fig.add_vline(x=cur, line=dict(color=C['yellow'], width=2, dash='dash'),
                  annotation=dict(text=f'NOW {cur:.4f}',
                                  font=dict(family='Share Tech Mono', size=9,
                                            color=C['yellow'])))
    lo = BASE_LAYOUT.copy()
    lo.update(dict(height=160, showlegend=False,
                   title=dict(text='VOL DISTRIBUTION',
                               font=dict(family='Share Tech Mono', size=9,
                                         color=C['muted']), x=0.01),
                   yaxis=dict(**BASE_LAYOUT['yaxis'], side='left')))
    fig.update_layout(**lo)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="main-title">⚡ GEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">GRAVITY EXPLOSION LAB</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">// INSTRUMENT</div>', unsafe_allow_html=True)
    TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD',
               'DOGE-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD']
    ticker_sel = st.selectbox('Ticker', TICKERS + ['Custom'], label_visibility='collapsed')
    if ticker_sel == 'Custom':
        ticker_sel = st.text_input('Enter ticker', value='BTC-USD',
                                   label_visibility='collapsed')
    ticker = ticker_sel

    st.markdown('<div class="section-hdr">// TIMEFRAME</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        period   = st.selectbox('Period',   ['7d','14d','30d','60d','90d'],
                                index=2, label_visibility='collapsed')
    with c2:
        interval = st.selectbox('Interval', ['1h','4h','1d'],
                                label_visibility='collapsed')

    st.markdown('<div class="section-hdr">// GRAVITY PARAMS</div>', unsafe_allow_html=True)
    n_levels = st.slider('Gravity Lines',    4, 12, 7)
    lookback = st.slider('Level Lookback',  50, 250, 120)

    st.markdown('<div class="section-hdr">// VOL PARAMS</div>', unsafe_allow_html=True)
    vol_window   = st.slider('Vol Window',    10, 40, 20)
    display_bars = st.slider('Display Bars', 60, 350, 160)

    st.markdown('<div class="section-hdr">// CONTROLS</div>', unsafe_allow_html=True)
    if st.button('⚡  SCAN NOW', use_container_width=True):
        st.cache_data.clear()
    auto_ref = st.checkbox('Auto-refresh (5 min)', value=False)
    if auto_ref:
        st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner('Fetching market data…'):
    df_raw = load_data(ticker, period, interval)

with st.spinner('Detecting gravity lines…'):
    levels = compute_gravity_lines(df_raw, n_levels=n_levels, lookback=lookback)
    touches = level_touch_count(df_raw, levels)

with st.spinner('Running volatility engine…'):
    df = compute_volatility(df_raw, window=vol_window)
    zones = get_compression_zones(df)

with st.spinner('Training breakout model…'):
    df_json    = df.to_json()
    levels_str = ','.join(str(l) for l in levels)
    bundle     = train_model(id(df), df_json, levels_str)

# ── Current snapshot ──────────────────────────────────────────────────────────
price_now     = float(df['Close'].iloc[-1])
vol_pct_now   = float(df['vol_pct'].fillna(0.5).iloc[-1])
is_comp       = bool(df['is_compressed'].fillna(False).iloc[-1])
is_expl       = bool(df['is_explosion'].fillna(False).iloc[-1])
bp            = predict_prob(df, levels, bundle)
nl            = nearest_levels(price_now, levels)
dist_pct_near = float(nl['nearest_pct'])
score, score_desc = setup_score(vol_pct_now, bp, dist_pct_near)
price_chg     = float(df['Close'].pct_change().iloc[-1])

# ── Signal label ──────────────────────────────────────────────────────────────
if is_expl:
    signal_html = '<span class="signal-pill pill-explosion">◈ VOLATILITY EXPLOSION DETECTED</span>'
elif is_comp and dist_pct_near < 0.025:
    signal_html = '<span class="signal-pill pill-aligned">◉ COMPRESSED AT GRAVITY LINE — HIGH CONFLUENCE</span>'
elif is_comp:
    signal_html = '<span class="signal-pill pill-compression">◉ VOLATILITY COMPRESSION PHASE</span>'
else:
    signal_html = '<span class="signal-pill pill-neutral">◌ NEUTRAL — NO SETUP</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f'<div class="main-title">GRAVITY EXPLOSION LAB</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="main-sub">VOLATILITY EXPLOSION TRACKER × CRYPTO GRAVITY LINES &nbsp;·&nbsp; {ticker}'
    f'&nbsp;·&nbsp; {interval} bars</div>',
    unsafe_allow_html=True,
)
st.markdown('<br>', unsafe_allow_html=True)

chart_col, metrics_col = st.columns([3, 1], gap='medium')

# ── LEFT: Charts ──────────────────────────────────────────────────────────────
with chart_col:
    st.markdown(signal_html, unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    fig_main = main_chart(df, levels, zones, display_bars, touches)
    st.plotly_chart(fig_main, use_container_width=True,
                    config={'displayModeBar': False})

    c_gauge, c_hist = st.columns([1, 1])
    with c_gauge:
        st.plotly_chart(vol_gauge_chart(vol_pct_now, bp),
                        use_container_width=True, config={'displayModeBar': False})
    with c_hist:
        st.plotly_chart(vol_histogram(df),
                        use_container_width=True, config={'displayModeBar': False})


# ── RIGHT: Metrics ────────────────────────────────────────────────────────────
with metrics_col:

    # Setup Score card
    if score >= 70:
        sc_grad = f'linear-gradient(135deg, #00ff88, #00d4ff)'
    elif score >= 45:
        sc_grad = f'linear-gradient(135deg, #ffcc00, #ff8c00)'
    else:
        sc_grad = f'linear-gradient(135deg, #ff3366, #b060ff)'

    st.markdown(f"""
    <div class="setup-score-wrap">
      <div class="setup-score-num"
           style="background:{sc_grad};-webkit-background-clip:text;
                  -webkit-text-fill-color:transparent;">{score}</div>
      <div class="setup-score-label">SETUP SCORE / 100</div>
      <div class="setup-score-desc" style="color:{'#00ff88' if score>=70 else '#ffcc00' if score>=45 else '#ff3366'};">
        {score_desc}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Price
    st.markdown('<div class="section-hdr">// PRICE</div>', unsafe_allow_html=True)
    ch_col = 'green' if price_chg >= 0 else 'red'
    st.markdown(f"""
    <div class="metric-card mc-{ch_col}">
      <div class="metric-label">Current Price</div>
      <div class="metric-value col-{ch_col}">{price_now:,.2f}</div>
      <div class="metric-sub">{'▲' if price_chg>=0 else '▼'} {abs(price_chg)*100:.3f}% last bar
           &nbsp;·&nbsp; {len(df)} bars loaded</div>
    </div>
    """, unsafe_allow_html=True)

    # Gravity Lines
    st.markdown('<div class="section-hdr">// GRAVITY LINES</div>', unsafe_allow_html=True)

    r_lv  = nl['resistance']['level']
    r_pct = nl['resistance']['pct']
    s_lv  = nl['support']['level']
    s_pct = nl['support']['pct']

    st.markdown(f"""
    <div class="metric-card mc-red">
      <div class="metric-label">Nearest Resistance</div>
      <div class="metric-value col-red">{r_lv:,.0f}</div>
      <div class="metric-sub">▲ {r_pct*100:.2f}% away
           &nbsp;·&nbsp; {touches.get(min(levels, key=lambda l: abs(l-r_lv)), 0)} touches</div>
    </div>
    <div class="metric-card mc-green">
      <div class="metric-label">Nearest Support</div>
      <div class="metric-value col-green">{s_lv:,.0f}</div>
      <div class="metric-sub">▼ {s_pct*100:.2f}% away
           &nbsp;·&nbsp; {touches.get(min(levels, key=lambda l: abs(l-s_lv)), 0)} touches</div>
    </div>
    """, unsafe_allow_html=True)

    # Volatility
    st.markdown('<div class="section-hdr">// VOLATILITY STATE</div>', unsafe_allow_html=True)
    vol_label  = 'COMPRESSED' if is_comp else ('EXPLODING' if is_expl else 'NORMAL')
    vol_mcolor = 'cyan' if is_comp else ('orange' if is_expl else 'yellow')
    vol_now    = float(df['vol'].iloc[-1])
    vol_z_now  = float(df['vol_z'].fillna(0).iloc[-1])

    st.markdown(f"""
    <div class="metric-card mc-{vol_mcolor}">
      <div class="metric-label">Vol Regime</div>
      <div class="metric-value col-{vol_mcolor}">{vol_label}</div>
      <div class="metric-sub">Percentile {vol_pct_now*100:.0f}th
           &nbsp;·&nbsp; Z {vol_z_now:+.2f}
           &nbsp;·&nbsp; σ {vol_now:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ML Signal
    st.markdown('<div class="section-hdr">// ML BREAKOUT SIGNAL</div>', unsafe_allow_html=True)
    ml_col  = 'green' if bp > 0.60 else ('yellow' if bp > 0.38 else 'red')
    ml_desc = 'HIGH PROBABILITY' if bp > 0.60 else ('MODERATE' if bp > 0.38 else 'LOW PROBABILITY')
    st.markdown(f"""
    <div class="metric-card mc-{ml_col}">
      <div class="metric-label">Breakout Probability</div>
      <div class="metric-value col-{ml_col}">{bp*100:.0f}%</div>
      <div class="metric-sub">{ml_desc} &nbsp;·&nbsp; GBM · 5-bar horizon</div>
    </div>
    """, unsafe_allow_html=True)

    # All levels
    st.markdown('<div class="section-hdr">// ALL GRAVITY LEVELS</div>', unsafe_allow_html=True)
    for lv in reversed(levels):
        is_r    = lv > price_now
        col     = '#ff3366' if is_r else '#00ff88'
        bg      = 'rgba(255,51,102,0.07)' if is_r else 'rgba(0,255,136,0.07)'
        border  = '#ff336633' if is_r else '#00ff8833'
        tag     = 'R' if is_r else 'S'
        dpct    = abs(price_now - lv) / price_now * 100
        tc      = touches.get(lv, 0)
        # proximity glow
        glow    = f'box-shadow: 0 0 6px {col}55;' if dpct < 1.5 else ''
        st.markdown(f"""
        <div class="level-row"
             style="background:{bg}; border-left:2px solid {col}; {glow}">
          <span style="color:{col};">{tag} &nbsp;{lv:,.0f}</span>
          <span style="color:#2a4a60;">{dpct:.1f}% · {tc}✕</span>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown('<br>', unsafe_allow_html=True)
    n_comp_zones = len(zones)
    n_expl_bars  = int(df['is_explosion'].sum())
    st.markdown(f"""
    <div style="font-family:Share Tech Mono;font-size:8px;color:#1e3a50;
                text-align:center;border-top:1px solid #0f2a40;padding-top:10px;
                line-height:1.8;">
      {n_comp_zones} compression zones detected<br>
      {n_expl_bars} explosion bars in window<br>
      tubakhxn × GEL v1.0 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)
