# ─────────────────────────────────────────────────────────────────────────────
#  GRAVITY EXPLOSION LAB  ⚡  v2.0
#  Volatility Explosion Tracker × Crypto Gravity Lines
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

st.set_page_config(
    page_title="GEL · Gravity Explosion Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #07090f;
    --surf:    #0d1117;
    --surf2:   #111827;
    --bord:    #1e2d3d;
    --bord2:   #253545;
    --text:    #dce8f0;
    --muted:   #4a6070;
    --dim:     #2a3a48;
    --green:   #10d98a;
    --red:     #f03358;
    --cyan:    #00c8f0;
    --yellow:  #f5c842;
    --orange:  #f07830;
    --purple:  #9060f0;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}

html, body, [class*="css"] { font-family:var(--sans); background:var(--bg); color:var(--text); }
.stApp { background:var(--bg); }
.stSpinner > div { border-top-color:var(--cyan) !important; }
.block-container { padding-top:1.2rem !important; }

section[data-testid="stSidebar"] {
    background:var(--surf);
    border-right:1px solid var(--bord);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p { color:var(--text) !important; }

.stSelectbox > div > div,
.stTextInput > div > div > input {
    background:var(--surf2) !important;
    border:1px solid var(--bord2) !important;
    border-radius:6px !important;
    color:var(--text) !important;
    font-family:var(--mono) !important;
    font-size:13px !important;
}
.stSlider > div > div > div { background:var(--cyan) !important; }
.stCheckbox > label { color:var(--muted) !important; font-size:13px; }

.stButton > button {
    background:linear-gradient(135deg,#0a1e30,#102840);
    border:1px solid rgba(0,200,240,0.30);
    color:var(--cyan);
    font-family:var(--mono);
    font-size:11px;
    letter-spacing:2px;
    border-radius:6px;
    padding:10px 0;
    width:100%;
    transition:all .2s ease;
}
.stButton > button:hover {
    border-color:var(--cyan);
    box-shadow:0 0 16px rgba(0,200,240,.20);
    transform:translateY(-1px);
}

.stTabs [data-baseweb="tab-list"] {
    background:var(--surf);
    border-bottom:1px solid var(--bord);
    gap:0; padding:0 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family:var(--mono); font-size:11px; letter-spacing:1.5px;
    color:var(--muted); padding:10px 22px;
    border-bottom:2px solid transparent; background:transparent;
}
.stTabs [aria-selected="true"] {
    color:var(--cyan) !important;
    border-bottom:2px solid var(--cyan) !important;
    background:transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top:18px; }

.streamlit-expanderHeader {
    font-family:var(--mono) !important; font-size:11px !important;
    letter-spacing:1px !important; color:var(--muted) !important;
    background:var(--surf) !important; border:1px solid var(--bord) !important;
    border-radius:6px !important;
}
.streamlit-expanderContent {
    background:var(--surf) !important; border:1px solid var(--bord) !important;
    border-top:none !important; border-radius:0 0 6px 6px !important;
}

/* ── Components ── */
.gel-header { display:flex; align-items:baseline; gap:10px; margin-bottom:3px; flex-wrap:wrap; }
.gel-wordmark { font-family:var(--mono); font-size:20px; font-weight:600; color:var(--cyan); letter-spacing:4px; }
.gel-badge {
    font-family:var(--mono); font-size:9px; letter-spacing:2px; color:var(--muted);
    background:var(--surf2); border:1px solid var(--bord); border-radius:20px; padding:2px 10px;
}
.gel-sub { font-family:var(--mono); font-size:9px; color:var(--dim); letter-spacing:2px; margin-bottom:14px; }

.sumbar { display:flex; gap:1px; background:var(--bord); border:1px solid var(--bord); border-radius:8px; overflow:hidden; margin-bottom:18px; }
.sumcell { flex:1; background:var(--surf); padding:10px 14px; }
.sumlabel { font-family:var(--mono); font-size:8px; letter-spacing:2px; color:var(--muted); text-transform:uppercase; }
.sumval { font-family:var(--mono); font-size:16px; font-weight:600; line-height:1.2; }
.sumsub { font-family:var(--mono); font-size:9px; color:var(--muted); }

.sigbanner { border-radius:8px; padding:12px 16px; margin-bottom:14px; display:flex; align-items:center; gap:12px; }
.sigicon { font-size:18px; }
.sigtitle { font-family:var(--mono); font-size:12px; font-weight:600; letter-spacing:1px; }
.sigdetail { font-family:var(--sans); font-size:12px; margin-top:2px; opacity:.75; }
.sig-expl { background:rgba(240,120,48,.12); border:1px solid rgba(240,120,48,.35); color:var(--orange); }
.sig-aln  { background:rgba(16,217,138,.10); border:1px solid rgba(16,217,138,.30); color:var(--green); }
.sig-comp { background:rgba(0,200,240,.10);  border:1px solid rgba(0,200,240,.28);  color:var(--cyan);  }
.sig-neut { background:rgba(74,96,112,.15);  border:1px solid rgba(74,96,112,.30);  color:var(--muted); }

.scorecard { text-align:center; padding:18px 12px 14px; background:var(--surf); border:1px solid var(--bord); border-radius:10px; margin-bottom:10px; position:relative; overflow:hidden; }
.scorecard::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:10px 10px 0 0; }
.sc-hi::before { background:linear-gradient(90deg,var(--green),var(--cyan)); }
.sc-md::before { background:linear-gradient(90deg,var(--yellow),var(--orange)); }
.sc-lo::before { background:linear-gradient(90deg,var(--red),var(--purple)); }
.scorenum { font-family:var(--mono); font-size:54px; font-weight:600; line-height:1; }
.scorelbl { font-family:var(--mono); font-size:8px; letter-spacing:3px; color:var(--muted); margin-top:3px; text-transform:uppercase; }
.scorebar { height:4px; background:var(--bord); border-radius:2px; margin:10px 0 8px; overflow:hidden; }
.scorefill { height:100%; border-radius:2px; }
.scoredesc { font-family:var(--sans); font-size:11px; color:var(--muted); line-height:1.4; }

.bkdown { display:flex; gap:4px; margin-bottom:10px; }
.bkpill { flex:1; text-align:center; border-radius:6px; padding:6px 4px; }
.bknum { font-family:var(--mono); font-size:16px; font-weight:600; }
.bklbl { font-family:var(--mono); font-size:7px; letter-spacing:1px; color:var(--muted); margin-top:2px; }

.mcard { background:var(--surf); border:1px solid var(--bord); border-radius:8px; padding:12px 14px; margin-bottom:8px; position:relative; overflow:hidden; }
.mcard::after { content:''; position:absolute; top:0; left:0; width:3px; height:100%; }
.mc-g::after { background:var(--green); }
.mc-r::after { background:var(--red); }
.mc-c::after { background:var(--cyan); }
.mc-y::after { background:var(--yellow); }
.mc-o::after { background:var(--orange); }
.mlbl { font-family:var(--mono); font-size:8px; letter-spacing:2px; color:var(--muted); text-transform:uppercase; margin-bottom:4px; }
.mval { font-family:var(--mono); font-size:22px; font-weight:600; line-height:1; }
.msub { font-family:var(--sans); font-size:11px; color:var(--muted); margin-top:3px; }
.mtip { font-family:var(--sans); font-size:10px; color:var(--dim); margin-top:6px; padding-top:6px; border-top:1px solid var(--bord); line-height:1.4; }

.shdr { font-family:var(--mono); font-size:8px; letter-spacing:3px; color:var(--dim); text-transform:uppercase; border-bottom:1px solid var(--bord); padding-bottom:5px; margin:14px 0 8px; }

.lvlrow { display:flex; justify-content:space-between; align-items:center; padding:6px 10px 6px 12px; margin-bottom:3px; border-radius:6px; font-family:var(--mono); font-size:11px; }
.lvlbadge { font-size:8px; letter-spacing:1px; padding:1px 5px; border-radius:3px; font-weight:600; }
.lvlmeta { font-size:9px; color:var(--muted); text-align:right; line-height:1.6; }

.slabel { font-family:var(--mono); font-size:9px; letter-spacing:2px; color:var(--muted); text-transform:uppercase; margin:12px 0 5px; display:block; }

.charthint { font-family:var(--sans); font-size:11px; color:var(--dim); margin-bottom:8px; display:flex; gap:16px; flex-wrap:wrap; }
.hintitem { display:flex; align-items:center; gap:5px; }
.hintdot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }

.infobox { background:rgba(0,200,240,.06); border:1px solid rgba(0,200,240,.18); border-radius:8px; padding:12px 14px; margin:10px 0; font-family:var(--sans); font-size:12px; color:var(--muted); line-height:1.6; }
.infobox strong { color:var(--text); }

.gel-footer { font-family:var(--mono); font-size:9px; color:var(--dim); text-align:center; border-top:1px solid var(--bord); padding-top:10px; margin-top:6px; line-height:2; }

.col-g { color:var(--green); } .col-r { color:var(--red); }
.col-c { color:var(--cyan);  } .col-y { color:var(--yellow); }
.col-o { color:var(--orange); }

#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker, period, interval):
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 60:
            raise ValueError("Not enough data")
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        return raw[['Open','High','Low','Close','Volume']].dropna()
    except Exception:
        return _mock()

def _mock():
    np.random.seed(99); n = 350
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='1h')
    p = [45000.0]
    for _ in range(n-1):
        s = np.random.choice([1,5], p=[0.95,0.05])
        p.append(p[-1]*(1+np.random.normal(0,.006*s)))
    p = np.array(p)
    return pd.DataFrame({'Open':p*(1+np.random.uniform(-.002,.002,n)),
        'High':p*(1+abs(np.random.uniform(0,.01,n))),
        'Low':p*(1-abs(np.random.uniform(0,.01,n))),
        'Close':p,'Volume':np.random.randint(300,6000,n).astype(float)},index=dates)

def compute_gravity_lines(df, n_levels=6, lookback=120):
    rec = df.tail(lookback)
    h,l,c = rec['High'].values, rec['Low'].values, rec['Close'].values
    pts = list(c)
    for i in range(2,len(rec)-2):
        if h[i]>h[i-1] and h[i]>h[i+1] and h[i]>h[i-2] and h[i]>h[i+2]: pts+=[h[i],h[i]]
        if l[i]<l[i-1] and l[i]<l[i+1] and l[i]<l[i-2] and l[i]<l[i+2]: pts+=[l[i],l[i]]
    n_levels = min(n_levels, len(set(pts))-1)
    if n_levels<2: return np.linspace(df['Low'].min(),df['High'].max(),6)
    km = KMeans(n_clusters=n_levels,n_init=15,random_state=42).fit(np.array(pts).reshape(-1,1))
    return np.sort(km.cluster_centers_.flatten())

def nearest_levels(price, levels):
    above = [(l,l-price) for l in levels if l>price]
    below = [(l,price-l) for l in levels if l<=price]
    r = (lambda a: {'level':a[0],'dist':a[1],'pct':a[1]/price})(min(above,key=lambda x:x[1])) if above else {'level':np.nan,'dist':np.nan,'pct':np.nan}
    s = (lambda b: {'level':b[0],'dist':b[1],'pct':b[1]/price})(min(below,key=lambda x:x[1])) if below else {'level':np.nan,'dist':np.nan,'pct':np.nan}
    dists = [x['pct'] for x in [r,s] if not np.isnan(x['pct'])]
    return {'resistance':r,'support':s,'nearest_pct':min(dists) if dists else 0.05}

def level_touch_count(df, levels, tol=0.004):
    c = df['Close'].values
    return {lv: int(np.sum(np.abs(c-lv)<lv*tol)) for lv in levels}

def compute_volatility(df, window=20):
    d = df.copy()
    d['ret'] = d['Close'].pct_change()
    d['vol'] = d['ret'].rolling(window).std() * np.sqrt(252*24)
    d['vol_pct'] = d['vol'].rolling(200,min_periods=window).rank(pct=True)
    d['is_compressed'] = d['vol_pct'] < 0.25
    d['is_explosion']  = (d['vol_pct']>0.80)&(d['vol']>d['vol'].shift(1))
    vm = d['vol'].rolling(150,min_periods=window).mean()
    vs = d['vol'].rolling(150,min_periods=window).std().replace(0,np.nan)
    d['vol_z'] = (d['vol']-vm)/vs
    d['atr'] = (d['High']-d['Low']).rolling(14).mean()
    d['atr_pct'] = (d['atr']/d['Close']).replace(0,np.nan)
    mid = d['Close'].rolling(window).mean()
    d['bb_width'] = 4*d['Close'].rolling(window).std()/mid
    return d

def get_compression_zones(df, min_bars=3):
    flags = df['is_compressed'].fillna(False)
    zones,in_z,start = [],[],None
    idx = list(df.index)
    for i,(ts,v) in enumerate(flags.items()):
        if v and not in_z: in_z,start=True,ts
        elif not v and in_z:
            in_z=False
            if (idx.index(ts)-idx.index(start))>=min_bars: zones.append((start,ts))
    return zones

@st.cache_data(ttl=600, show_spinner=False)
def train_model(_h, df_json, levels_str):
    df = pd.read_json(StringIO(df_json))
    lvs = np.array([float(x) for x in levels_str.split(',')])
    d = df.copy().dropna()
    d['dist_nearest'] = d['Close'].apply(lambda p: min(abs(p-lv)/p for lv in lvs))
    d['vol_trend'] = d['vol'].diff(5); d['vol_accel'] = d['vol_trend'].diff(3)
    d['price_mom'] = d['Close'].pct_change(5); d['atr_trend'] = d['atr_pct'].diff(5)
    fv = d['vol'].shift(-5)
    d['label'] = ((fv/(d['vol']+1e-9))>1.5).astype(int)
    FEATS = ['vol_pct','vol_z','dist_nearest','vol_trend','vol_accel','price_mom','atr_pct','atr_trend','bb_width']
    v = d[FEATS+['label']].dropna()
    if len(v)<60 or len(v['label'].unique())<2: return None
    X,y = v[FEATS].values, v['label'].values
    sp = int(len(X)*.80)
    if len(np.unique(y[:sp]))<2: return None
    sc = StandardScaler().fit(X[:sp])
    clf = GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=.08,subsample=.8,random_state=42).fit(sc.transform(X[:sp]),y[:sp])
    return {'clf':clf,'sc':sc,'feats':FEATS}

def predict_prob(df, levels, bundle):
    if not bundle: return 0.5
    clf,sc,FEATS = bundle['clf'],bundle['sc'],bundle['feats']
    row = df.tail(1).copy()
    row['dist_nearest'] = row['Close'].apply(lambda p: min(abs(p-lv)/p for lv in levels))
    row['vol_trend'] = df['vol'].diff(5).iloc[-1]; row['vol_accel'] = df['vol'].diff(5).diff(3).iloc[-1]
    row['price_mom'] = df['Close'].pct_change(5).iloc[-1]; row['atr_trend'] = df['atr_pct'].diff(5).iloc[-1]
    try:
        x = row[FEATS].values
        return 0.5 if np.any(np.isnan(x)) else float(clf.predict_proba(sc.transform(x))[0][1])
    except: return 0.5

def setup_score(vol_pct, bp, dist_pct):
    comp = max(0.,(0.30-vol_pct)/0.30)*40
    prox = max(0.,(0.03-dist_pct)/0.03)*30
    ml   = bp*30
    sc   = int(min(100,comp+prox+ml))
    desc = ("Strong confluence — all three conditions align" if sc>=70 else
            "Setup developing — watch for a trigger" if sc>=50 else
            "Early stage — insufficient confluence yet" if sc>=30 else
            "No setup — vol expanded or far from levels")
    return sc, desc


# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

C = {'bg':'#07090f','surf':'#0d1117','grid':'#111827','muted':'#4a6070','text':'#dce8f0',
     'green':'#10d98a','red':'#f03358','cyan':'#00c8f0','yellow':'#f5c842','orange':'#f07830','purple':'#9060f0'}

BL = dict(paper_bgcolor=C['bg'],plot_bgcolor=C['surf'],
          font=dict(family='IBM Plex Mono',color=C['muted'],size=10),
          margin=dict(l=8,r=60,t=10,b=8),hovermode='x unified',
          legend=dict(bgcolor='rgba(0,0,0,0)',orientation='h',font=dict(size=9,color=C['muted']),yanchor='bottom',y=1.01,x=0),
          xaxis=dict(gridcolor=C['grid'],zerolinecolor=C['grid'],showgrid=True,color=C['muted'],rangeslider=dict(visible=False)),
          yaxis=dict(gridcolor=C['grid'],zerolinecolor=C['grid'],showgrid=True,color=C['muted'],side='right'))

def main_chart(df,levels,zones,n_bars,touches,price_now):
    plot = df.tail(n_bars)
    fig  = make_subplots(rows=2,cols=1,row_heights=[0.72,.28],shared_xaxes=True,vertical_spacing=.02)
    for z0,z1 in zones:
        if z1<plot.index[0]: continue
        for row in (1,2):
            fig.add_vrect(x0=max(z0,plot.index[0]),x1=min(z1,plot.index[-1]),row=row,col=1,
                          fillcolor='rgba(0,200,240,0.04)',line_width=0)
    fig.add_trace(go.Candlestick(x=plot.index,open=plot['Open'],high=plot['High'],low=plot['Low'],close=plot['Close'],
        increasing=dict(line=dict(color=C['green'],width=1),fillcolor='rgba(16,217,138,0.28)'),
        decreasing=dict(line=dict(color=C['red'],width=1),fillcolor='rgba(240,51,88,0.28)'),
        name='Price',whiskerwidth=.3),row=1,col=1)
    lo,hi = plot['Low'].min(),plot['High'].max(); span=hi-lo or 1
    vis_lo,vis_hi = lo-span*.06,hi+span*.06
    max_tc = max(touches.values()) if touches else 1
    for lv in levels:
        if not (vis_lo<=lv<=vis_hi): continue
        is_r = lv>price_now; col=C['red'] if is_r else C['green']
        dp = abs(price_now-lv)/price_now; prox=max(0.,1.-dp/.04)
        tw = touches.get(lv,0)/(max_tc or 1)
        fig.add_hline(y=lv,row=1,col=1,line=dict(color=col,width=.8+prox*1.6+tw*.8,dash='dot'),opacity=min(1.,.20+prox*.55+tw*.15))
        fig.add_annotation(x=plot.index[-1],y=lv,xanchor='left',showarrow=False,
            text=f" {'R' if is_r else 'S'} {lv:,.0f}",font=dict(family='IBM Plex Mono',size=9,color=col),row=1,col=1)
    expl = plot[plot['is_explosion']==True]
    if not expl.empty:
        fig.add_trace(go.Scatter(x=expl.index,y=expl['High']*1.007,mode='markers',
            marker=dict(symbol='triangle-up',size=10,color=C['orange'],line=dict(color=C['orange'],width=1)),
            name='Explosion'),row=1,col=1)
    fig.add_trace(go.Scatter(x=plot.index,y=plot['vol'],mode='lines',name='Volatility',
        line=dict(color=C['cyan'],width=1.5),fill='tozeroy',fillcolor='rgba(0,200,240,0.06)'),row=2,col=1)
    if not expl.empty:
        fig.add_trace(go.Scatter(x=expl.index,y=expl['vol'],mode='markers',showlegend=False,
            marker=dict(symbol='circle',size=6,color=C['orange'])),row=2,col=1)
    lo2 = BL.copy()
    lo2.update(dict(height=510,showlegend=True,
        xaxis2=dict(gridcolor=C['grid'],zerolinecolor=C['grid'],showgrid=True,color=C['muted']),
        yaxis2=dict(gridcolor=C['grid'],zerolinecolor=C['grid'],showgrid=True,color=C['muted'],side='right',tickformat='.3f')))
    fig.update_layout(**lo2)
    return fig

def vol_gauge(vol_pct,bp):
    fig = go.Figure()
    fig.add_trace(go.Indicator(mode='gauge+number',value=round(vol_pct*100,1),
        number=dict(suffix='th',font=dict(family='IBM Plex Mono',size=26,color=C['cyan'])),
        title=dict(text='VOL PERCENTILE',font=dict(family='IBM Plex Mono',size=9,color=C['muted'])),
        gauge=dict(axis=dict(range=[0,100],tickfont=dict(size=8,color=C['muted']),tickcolor=C['muted'],dtick=25),
            bar=dict(color=C['cyan'],thickness=.22),bgcolor=C['surf'],bordercolor=C['grid'],borderwidth=1,
            steps=[dict(range=[0,25],color='rgba(0,200,240,.12)'),dict(range=[25,60],color='rgba(74,96,112,.08)'),
                   dict(range=[60,80],color='rgba(245,200,66,.07)'),dict(range=[80,100],color='rgba(240,51,88,.12)')],
            threshold=dict(line=dict(color=C['orange'],width=2),thickness=.7,value=bp*100)),
        domain=dict(x=[0,1],y=[0,1])))
    fig.update_layout(paper_bgcolor=C['bg'],plot_bgcolor=C['surf'],height=175,
        margin=dict(l=10,r=10,t=28,b=8),font=dict(family='IBM Plex Mono',color=C['text']))
    return fig

def vol_histogram(df):
    vol = df['vol'].dropna(); cur = vol.iloc[-1]; pct = float(df['vol_pct'].iloc[-1])
    bins = np.linspace(vol.min(),vol.max(),41); counts,edges = np.histogram(vol.values,bins=bins)
    step = edges[1]-edges[0]
    colors = ['rgba(0,200,240,.45)' if (vol<=e+step).mean()<.25 else
              'rgba(240,51,88,.40)' if (vol<=e+step).mean()>.80 else
              'rgba(74,96,112,.30)' for e in edges[:-1]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=edges[:-1],y=counts,width=step,marker_color=colors,name=''))
    fig.add_vline(x=cur,line=dict(color=C['yellow'],width=2,dash='dash'),
        annotation=dict(text=f' {pct*100:.0f}th',font=dict(family='IBM Plex Mono',size=9,color=C['yellow']),xanchor='left'))
    lo = BL.copy()
    lo.update(dict(height=175,showlegend=False,
        title=dict(text='VOL DISTRIBUTION',font=dict(family='IBM Plex Mono',size=9,color=C['muted']),x=.01),
        yaxis={**BL['yaxis'],'side':'left','showgrid':False,'showticklabels':False},
        xaxis={**BL['xaxis'],'showgrid':False,'tickformat':'.3f'}))
    fig.update_layout(**lo)
    return fig

def levels_chart(df,levels,touches,price_now):
    fig = go.Figure(); max_tc=max(touches.values()) if touches else 1
    for lv in sorted(levels,reverse=True):
        is_r=lv>price_now; col=C['red'] if is_r else C['green']
        dpct=(lv-price_now)/price_now*100; tc=touches.get(lv,0)
        fig.add_trace(go.Bar(x=[dpct],y=[f"{'R' if is_r else 'S'} {lv:,.0f}"],
            orientation='h',marker=dict(color=col,opacity=.25+tc/(max_tc or 1)*.6),width=.6,
            showlegend=False,name='',
            hovertemplate=f"<b>{'Resistance' if is_r else 'Support'}</b><br>Level:{lv:,.2f}<br>Dist:{abs(dpct):.2f}%<br>Touches:{tc}<extra></extra>"))
    fig.add_vline(x=0,line=dict(color=C['yellow'],width=1.5),
        annotation=dict(text=' NOW',font=dict(size=9,color=C['yellow'],family='IBM Plex Mono'),xanchor='left'))
    lo2=BL.copy()
    lo2.update(dict(height=max(200,len(levels)*38+40),showlegend=False,
        title=dict(text='DISTANCE FROM PRICE (%)',font=dict(family='IBM Plex Mono',size=9,color=C['muted']),x=.0),
        xaxis={**BL['xaxis'],'ticksuffix':'%','zeroline':False},
        yaxis={**BL['yaxis'],'side':'left','showgrid':False},bargap=.3))
    fig.update_layout(**lo2)
    return fig

def vol_timeline(df,n_bars=200):
    plot=df.tail(n_bars).copy()
    fig=make_subplots(rows=3,cols=1,row_heights=[.44,.32,.24],shared_xaxes=True,vertical_spacing=.02)
    fig.add_trace(go.Scatter(x=plot.index,y=plot['Close'],mode='lines',line=dict(color=C['text'],width=1),
        name='Price',fill='tozeroy',fillcolor='rgba(220,232,240,.03)'),row=1,col=1)
    fig.add_trace(go.Scatter(x=plot.index,y=plot['vol'],mode='lines',line=dict(color=C['cyan'],width=1.5),
        name='Vol',fill='tozeroy',fillcolor='rgba(0,200,240,.07)'),row=2,col=1)
    comp=plot[plot['is_compressed']==True]
    if not comp.empty:
        fig.add_trace(go.Scatter(x=comp.index,y=comp['vol'],mode='markers',
            marker=dict(color=C['cyan'],size=5,symbol='circle'),name='Compressed'),row=2,col=1)
    expl=plot[plot['is_explosion']==True]
    if not expl.empty:
        fig.add_trace(go.Scatter(x=expl.index,y=expl['vol'],mode='markers',
            marker=dict(color=C['orange'],size=7,symbol='diamond'),name='Explosion'),row=2,col=1)
    fig.add_trace(go.Scatter(x=plot.index,y=plot['bb_width'],mode='lines',
        line=dict(color=C['purple'],width=1.2),name='BB Width',
        fill='tozeroy',fillcolor='rgba(144,96,240,.07)'),row=3,col=1)
    lo2=BL.copy()
    lo2.update(dict(height=510,showlegend=True,
        xaxis2=dict(gridcolor=C['grid'],zerolinecolor=C['grid'],showgrid=True,color=C['muted']),
        xaxis3=dict(gridcolor=C['grid'],zerolinecolor=C['grid'],showgrid=True,color=C['muted']),
        yaxis2=dict(gridcolor=C['grid'],zerolinecolor=C['grid'],showgrid=True,color=C['muted'],side='right',tickformat='.3f'),
        yaxis3=dict(gridcolor=C['grid'],zerolinecolor=C['grid'],showgrid=True,color=C['muted'],side='right')))
    fig.update_layout(**lo2)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    'Quick': dict(period='7d', interval='1h', n_levels=6, lookback=100, vol_window=20, display_bars=120),
    'Swing': dict(period='30d',interval='4h', n_levels=8, lookback=150, vol_window=20, display_bars=140),
    'Daily': dict(period='90d',interval='1d', n_levels=10,lookback=200, vol_window=14, display_bars=90),
    'Deep':  dict(period='60d',interval='2h', n_levels=7, lookback=200, vol_window=20, display_bars=160),
}
TICKERS = ['BTC-USD','ETH-USD','SOL-USD','BNB-USD','DOGE-USD','XRP-USD','ADA-USD','AVAX-USD','MATIC-USD','LINK-USD']

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:600;color:#00c8f0;letter-spacing:4px;">⚡ GEL</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;letter-spacing:2px;color:#2a3a48;margin-top:2px;">GRAVITY EXPLOSION LAB v2.0</div>
    </div>""", unsafe_allow_html=True)

    # Presets
    st.markdown('<span class="slabel">PRESET MODE</span>', unsafe_allow_html=True)
    active = st.session_state.get('preset','Swing')
    pcols = st.columns(4)
    for i,(name,cfg) in enumerate(PRESETS.items()):
        with pcols[i]:
            btn_style = "border-color:var(--cyan)!important;" if name==active else ""
            if st.button(name,key=f'p_{name}',use_container_width=True):
                st.session_state['preset']=name
                for k,v in cfg.items(): st.session_state[k]=v
                active=name; st.rerun()
    cfg = PRESETS[active]

    st.markdown('<span class="slabel">INSTRUMENT</span>', unsafe_allow_html=True)
    tsel = st.selectbox('Ticker',TICKERS+['Custom'],label_visibility='collapsed')
    ticker = st.text_input('Custom',value='BTC-USD',label_visibility='collapsed') if tsel=='Custom' else tsel

    st.markdown('<span class="slabel">TIMEFRAME</span>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    periods = ['7d','14d','30d','60d','90d']
    intervals = ['1h','2h','4h','1d']
    with c1:
        period = st.selectbox('Period',periods,
            index=periods.index(st.session_state.get('period',cfg['period'])),
            label_visibility='collapsed')
    with c2:
        interval = st.selectbox('Interval',intervals,
            index=intervals.index(st.session_state.get('interval',cfg['interval'])),
            label_visibility='collapsed')

    st.markdown('<span class="slabel">GRAVITY LINES</span>', unsafe_allow_html=True)
    n_levels = st.slider('Number of levels',4,12,st.session_state.get('n_levels',cfg['n_levels']),
        help='How many support/resistance clusters to detect')
    lookback = st.slider('Lookback (bars)',50,250,st.session_state.get('lookback',cfg['lookback']),
        help='How far back to scan for swing highs and lows')

    st.markdown('<span class="slabel">VOLATILITY</span>', unsafe_allow_html=True)
    vol_window   = st.slider('Vol window',10,40,st.session_state.get('vol_window',cfg['vol_window']),
        help='Rolling window size for volatility calculation')
    display_bars = st.slider('Chart bars',60,350,st.session_state.get('display_bars',cfg['display_bars']),
        help='Number of recent bars shown on chart')

    st.markdown('<span class="slabel">CONTROLS</span>', unsafe_allow_html=True)
    if st.button('⚡  SCAN NOW',use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.checkbox('Auto-refresh every 5 min',value=False)

    with st.expander('ℹ️  HOW TO READ THIS'):
        st.markdown("""
**Setup Score (0–100)**
Combines 3 signals into one number:
- 🔵 **Compression** (40 pts) — is vol in the bottom 25%?
- 🟢 **Gravity proximity** (30 pts) — is price near a key level?
- 🤖 **ML signal** (30 pts) — does the model agree?

**Gravity Lines**
Clustered swing highs/lows from recent price history. Brighter line = more touches = stronger level.

**Compression zones (cyan shading)**
Bars where vol was historically low. Breakouts often follow these.

**Orange triangles ▲**
Vol spiked above the 80th percentile and was still rising.

**Level Map tab**
Shows all levels as horizontal bars. Wider bar = more touches = stronger.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner('Fetching market data…'):
    df_raw = load_data(ticker, period, interval)
is_mock = df_raw.index[-1].year < 2020

with st.spinner('Detecting gravity lines…'):
    levels  = compute_gravity_lines(df_raw, n_levels=n_levels, lookback=lookback)
    touches = level_touch_count(df_raw, levels)

with st.spinner('Running volatility engine…'):
    df    = compute_volatility(df_raw, window=vol_window)
    zones = get_compression_zones(df)

with st.spinner('Training ML model…'):
    bundle = train_model(id(df), df.to_json(), ','.join(str(l) for l in levels))

# Snapshot
price_now   = float(df['Close'].iloc[-1])
price_prev  = float(df['Close'].iloc[-2])
price_chg   = (price_now-price_prev)/price_prev
vol_pct_now = float(df['vol_pct'].fillna(.5).iloc[-1])
vol_now     = float(df['vol'].fillna(0).iloc[-1])
vol_z_now   = float(df['vol_z'].fillna(0).iloc[-1])
is_comp     = bool(df['is_compressed'].fillna(False).iloc[-1])
is_expl     = bool(df['is_explosion'].fillna(False).iloc[-1])
bp          = predict_prob(df, levels, bundle)
nl          = nearest_levels(price_now, levels)
dist_near   = float(nl['nearest_pct'])
score, score_desc = setup_score(vol_pct_now, bp, dist_near)
atr_now     = float(df['atr_pct'].fillna(0).iloc[-1])
n_czones    = len(zones)
n_ebars     = int(df['is_explosion'].sum())


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(f"""
<div class="gel-header">
  <span class="gel-wordmark">⚡ GEL</span>
  <span class="gel-badge">v2.0</span>
  <span class="gel-badge">{ticker}</span>
  <span class="gel-badge">{interval} · {period}</span>
  {'<span class="gel-badge" style="color:#f5c842;border-color:#f5c84255;">⚠ DEMO DATA</span>' if is_mock else ''}
</div>
<div class="gel-sub">GRAVITY EXPLOSION LAB · {pd.Timestamp.now().strftime('%d %b %Y %H:%M')} UTC</div>
""", unsafe_allow_html=True)

# Summary bar
arrow    = '▲' if price_chg>=0 else '▼'
pc_col   = '#10d98a' if price_chg>=0 else '#f03358'
vs       = 'COMPRESSED' if is_comp else ('EXPLODING' if is_expl else 'NORMAL')
vs_col   = '#00c8f0' if is_comp else ('#f07830' if is_expl else '#f5c842')
bp_col   = '#10d98a' if bp>.60 else ('#f5c842' if bp>.38 else '#f03358')
sc_col   = '#10d98a' if score>=70 else ('#f5c842' if score>=45 else '#f03358')

st.markdown(f"""
<div class="sumbar">
  <div class="sumcell">
    <div class="sumlabel">Last Price</div>
    <div class="sumval" style="color:{pc_col}">{price_now:,.2f}</div>
    <div class="sumsub">{arrow} {abs(price_chg)*100:.3f}% prev bar</div>
  </div>
  <div class="sumcell">
    <div class="sumlabel">Vol State</div>
    <div class="sumval" style="color:{vs_col};font-size:13px;padding-top:3px">{vs}</div>
    <div class="sumsub">{vol_pct_now*100:.0f}th pct · z{vol_z_now:+.1f}</div>
  </div>
  <div class="sumcell">
    <div class="sumlabel">ML Breakout Prob</div>
    <div class="sumval" style="color:{bp_col}">{bp*100:.0f}%</div>
    <div class="sumsub">5-bar horizon · GBM</div>
  </div>
  <div class="sumcell">
    <div class="sumlabel">Setup Score</div>
    <div class="sumval" style="color:{sc_col}">{score}<span style="font-size:13px;color:#2a3a48">/100</span></div>
    <div class="sumsub">{score_desc[:34]}…</div>
  </div>
  <div class="sumcell">
    <div class="sumlabel">Activity</div>
    <div class="sumval" style="color:#dce8f0">{n_levels}</div>
    <div class="sumsub">{n_czones} compression zones · {n_ebars} explosions</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Signal banner
if is_expl:
    sc2,si,st2,sd = 'sig-expl','💥','VOLATILITY EXPLOSION DETECTED',f'Vol at {vol_pct_now*100:.0f}th pct and rising. High-momentum conditions active.'
elif is_comp and dist_near<.025:
    sc2,si,st2,sd = 'sig-aln','🎯','COMPRESSED AT GRAVITY LINE — HIGH CONFLUENCE',f'Vol compressed + price within {dist_near*100:.1f}% of a key level. Classic pre-breakout setup.'
elif is_comp:
    sc2,si,st2,sd = 'sig-comp','🔵','VOLATILITY COMPRESSION PHASE',f'Vol at {vol_pct_now*100:.0f}th pct. Energy building — watch for a gravity line touch.'
else:
    sc2,si,st2,sd = 'sig-neut','○','NEUTRAL — NO SETUP','No actionable confluence. Vol expanded or price is far from key levels.'

st.markdown(f"""
<div class="sigbanner {sc2}">
  <div class="sigicon">{si}</div>
  <div>
    <div class="sigtitle">{st2}</div>
    <div class="sigdetail">{sd}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Main layout
chart_col, metrics_col = st.columns([3,1], gap='medium')

with chart_col:
    tab1, tab2, tab3 = st.tabs(['  📈  PRICE & LEVELS  ','  📊  VOLATILITY DEEP DIVE  ','  🗺  LEVEL MAP  '])

    with tab1:
        st.markdown("""
        <div class="charthint">
          <span class="hintitem"><span class="hintdot" style="background:#00c8f0;opacity:.5"></span>Compression zone</span>
          <span class="hintitem"><span class="hintdot" style="background:#f03358"></span>Resistance</span>
          <span class="hintitem"><span class="hintdot" style="background:#10d98a"></span>Support</span>
          <span class="hintitem"><span class="hintdot" style="background:#f07830"></span>Explosion bar</span>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(main_chart(df,levels,zones,display_bars,touches,price_now),
                        use_container_width=True, config={'displayModeBar':False})
        g1,g2 = st.columns(2)
        with g1: st.plotly_chart(vol_gauge(vol_pct_now,bp),use_container_width=True,config={'displayModeBar':False})
        with g2: st.plotly_chart(vol_histogram(df),use_container_width=True,config={'displayModeBar':False})

    with tab2:
        st.plotly_chart(vol_timeline(df,n_bars=min(display_bars,300)),
                        use_container_width=True, config={'displayModeBar':False})
        with st.expander('ℹ️  Reading this chart'):
            st.markdown("""
- **Top panel** — price action
- **Middle panel** — rolling annualized volatility. 🔵 Cyan dots = compressed bars. 🟠 Orange diamonds = explosion bars.
- **Bottom panel** — Bollinger Band width (secondary compression signal). Narrow = energy coiling. Wide = already expanded.

Breakouts tend to follow long narrow BB width periods. Look for cyan dots *right before* a sharp price move.
            """)

    with tab3:
        st.plotly_chart(levels_chart(df,levels,touches,price_now),
                        use_container_width=True, config={'displayModeBar':False})
        st.markdown("""
        <div class="infobox">
          <strong>How to read:</strong> Each bar is a gravity level. Bar width = touch count — wider means the level has been tested more times and is historically stronger.
          <span style="color:#f03358">Red bars (left of zero) = resistance</span> above price.
          <span style="color:#10d98a">Green bars (right of zero) = support</span> below price.
          The yellow line is your current price.
        </div>""", unsafe_allow_html=True)

with metrics_col:

    # Score card
    if score>=70: sc_c,sc_cl,bar_c='#10d98a','sc-hi','linear-gradient(90deg,#10d98a,#00c8f0)'
    elif score>=45: sc_c,sc_cl,bar_c='#f5c842','sc-md','linear-gradient(90deg,#f5c842,#f07830)'
    else: sc_c,sc_cl,bar_c='#f03358','sc-lo','linear-gradient(90deg,#f03358,#9060f0)'

    st.markdown(f"""
    <div class="scorecard {sc_cl}">
      <div class="scorenum" style="color:{sc_c}">{score}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#2a3a48">/100</div>
      <div class="scorelbl">SETUP SCORE</div>
      <div class="scorebar"><div class="scorefill" style="width:{score}%;background:{bar_c}"></div></div>
      <div class="scoredesc">{score_desc}</div>
    </div>""", unsafe_allow_html=True)

    # Score breakdown
    cp = int(max(0.,(0.30-vol_pct_now)/0.30)*40)
    pp = int(max(0.,(0.03-dist_near)/0.03)*30)
    mp = int(bp*30)
    st.markdown(f"""
    <div class="bkdown">
      <div class="bkpill" style="background:rgba(0,200,240,.08);border:1px solid rgba(0,200,240,.2)">
        <div class="bknum" style="color:#00c8f0">{cp}</div>
        <div class="bklbl">COMPRESS</div>
      </div>
      <div class="bkpill" style="background:rgba(16,217,138,.08);border:1px solid rgba(16,217,138,.2)">
        <div class="bknum" style="color:#10d98a">{pp}</div>
        <div class="bklbl">GRAVITY</div>
      </div>
      <div class="bkpill" style="background:rgba(144,96,240,.08);border:1px solid rgba(144,96,240,.2)">
        <div class="bknum" style="color:#9060f0">{mp}</div>
        <div class="bklbl">ML</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Key levels
    st.markdown('<div class="shdr">KEY LEVELS</div>', unsafe_allow_html=True)
    r_lv,r_pct = nl['resistance']['level'],nl['resistance']['pct']
    s_lv,s_pct = nl['support']['level'],nl['support']['pct']
    r_tc = touches.get(min(levels,key=lambda l:abs(l-r_lv)),0) if not np.isnan(r_lv) else 0
    s_tc = touches.get(min(levels,key=lambda l:abs(l-s_lv)),0) if not np.isnan(s_lv) else 0
    st.markdown(f"""
    <div class="mcard mc-r">
      <div class="mlbl">Nearest Resistance</div>
      <div class="mval col-r">{r_lv:,.0f}</div>
      <div class="msub">▲ {r_pct*100:.2f}% away · {r_tc} touches</div>
    </div>
    <div class="mcard mc-g">
      <div class="mlbl">Nearest Support</div>
      <div class="mval col-g">{s_lv:,.0f}</div>
      <div class="msub">▼ {s_pct*100:.2f}% away · {s_tc} touches</div>
    </div>""", unsafe_allow_html=True)

    # Volatility state
    st.markdown('<div class="shdr">VOLATILITY STATE</div>', unsafe_allow_html=True)
    vm,vmc = ('c','cyan') if is_comp else (('o','orange') if is_expl else ('y','yellow'))
    tip = ('🔵 Vol is coiling — best setups form here. Watch for a gravity line touch.' if is_comp else
           '🟠 Vol expanding rapidly — momentum is high but chasing is risky.' if is_expl else
           '🟡 Vol in normal range — no extreme compression or expansion.')
    st.markdown(f"""
    <div class="mcard mc-{vm}">
      <div class="mlbl">Regime</div>
      <div class="mval col-{vmc}">{vs}</div>
      <div class="msub">{vol_pct_now*100:.0f}th pct · z{vol_z_now:+.2f} · σ={vol_now:.4f}</div>
      <div class="mtip">{tip}</div>
    </div>""", unsafe_allow_html=True)

    # ML signal
    st.markdown('<div class="shdr">ML SIGNAL</div>', unsafe_allow_html=True)
    ml_m = 'g' if bp>.60 else ('y' if bp>.38 else 'r')
    ml_c = 'g' if bp>.60 else ('y' if bp>.38 else 'r')
    ml_l = 'HIGH' if bp>.60 else ('MODERATE' if bp>.38 else 'LOW')
    st.markdown(f"""
    <div class="mcard mc-{ml_m}">
      <div class="mlbl">Breakout Probability</div>
      <div class="mval col-{ml_c}">{bp*100:.0f}%</div>
      <div class="msub">{ml_l} · GBM · 5-bar horizon</div>
      <div class="mtip">P(vol surges >50% in 5 bars). Uses compression, ATR, momentum & gravity proximity.
      {'⚠ Model fallback (50%) — low data variance.' if not bundle else ''}</div>
    </div>""", unsafe_allow_html=True)

    # ATR
    st.markdown('<div class="shdr">RANGE</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mcard mc-y">
      <div class="mlbl">ATR % of Price</div>
      <div class="mval col-y">{atr_now*100:.2f}%</div>
      <div class="msub">Avg true range · 14-bar</div>
    </div>""", unsafe_allow_html=True)

    # All levels
    st.markdown('<div class="shdr">ALL LEVELS</div>', unsafe_allow_html=True)
    max_tc = max(touches.values()) if touches else 1
    for lv in reversed(levels):
        is_r = lv>price_now
        col  = '#f03358' if is_r else '#10d98a'
        bg   = 'rgba(240,51,88,.07)' if is_r else 'rgba(16,217,138,.07)'
        tag  = 'R' if is_r else 'S'
        dpct = abs(price_now-lv)/price_now*100
        tc   = touches.get(lv,0)
        glow = f'box-shadow:0 0 8px {col}44;' if dpct<1.5 else ''
        bw   = int(tc/(max_tc or 1)*36)
        st.markdown(f"""
        <div class="lvlrow" style="background:{bg};border-left:3px solid {col};{glow}">
          <div>
            <span class="lvlbadge" style="background:{col}22;color:{col}">{tag}</span>
            <span style="color:{col};margin-left:6px">{lv:,.0f}</span>
          </div>
          <div class="lvlmeta">
            {dpct:.1f}%<br>
            <span style="font-size:7px;letter-spacing:0;color:#1e2d3d">{'█'*bw}{'░'*(36-bw)}</span> {tc}✕
          </div>
        </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="gel-footer">
      {n_czones} compression zones · {n_ebars} explosions<br>
      {len(df)} bars · {interval} · {period}<br>
      tubakhxn × GEL v2.0
    </div>""", unsafe_allow_html=True)
