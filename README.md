# ⚡ Gravity Explosion Lab (GEL)

**Volatility Explosion Tracker × Crypto Gravity Lines**

A combined Streamlit dashboard that merges two analytical frameworks into one unified signal engine: *where* price is likely to react (gravity lines) and *when* it's about to move (volatility explosion detection).

> Built by [tubakhxn](https://github.com/tubakhxn) · Combined & extended with Claude

---

## What It Does

Most crypto traders use support/resistance and volatility separately. GEL combines them into a single **Setup Score (0–100)** that only fires high when all three conditions align simultaneously:

1. **Volatility is compressed** — price is coiling, energy is building
2. **Price is near a gravity line** — a historically significant level where price has repeatedly reacted
3. **The ML model agrees** — a Gradient Boosted Classifier trained on both signal sets confirms elevated breakout probability

When all three align, the dashboard flags it as a **HIGH CONFLUENCE** setup.

---

## Source Projects

| Repo | What it contributed |
|---|---|
| [Volatility-Explosion-Tracker](https://github.com/tubakhxn/Volatility-Explosion-Tracker) | Rolling vol analytics, compression/explosion detection, GBM breakout predictor |
| [Crypto-Gravity-Lines](https://github.com/tubakhxn/Crypto-Gravity-Lines) | KMeans level detection, touch-count weighting, proximity scoring |

---

## Features

**Charts**
- Candlestick price chart with gravity lines overlaid (brightness and width scale with proximity and historical significance)
- Compression zones shaded in cyan across both the price and volatility subplots
- Orange triangle markers on volatility explosion bars
- Vol percentile gauge showing where current vol sits in its historical range
- Vol distribution histogram with a "NOW" marker

**Metrics Panel**
- Setup Score with color-coded gradient (green → yellow → red)
- Current price and last-bar change
- Nearest resistance and nearest support with distance % and touch count
- Volatility regime (COMPRESSED / EXPLODING / NORMAL) with raw vol, percentile, and z-score
- ML breakout probability with model metadata
- Full level table with S/R tags, distance %, and touch counts — proximity glow on near levels

**Signals**
- `◈ VOLATILITY EXPLOSION DETECTED` — vol in top 20th percentile and rising
- `◉ COMPRESSED AT GRAVITY LINE — HIGH CONFLUENCE` — compression + proximity together
- `◉ VOLATILITY COMPRESSION PHASE` — compression only
- `◌ NEUTRAL — NO SETUP` — no actionable conditions

---

## How to Run

**1. Clone or download**
```bash
git clone https://github.com/tubakhxn/Volatility-Explosion-Tracker
# or just download app.py and requirements.txt
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## Requirements

```
streamlit>=1.32.0
yfinance>=0.2.38
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
scikit-learn>=1.4.0
```

Python 3.10+ recommended.

---

## Sidebar Controls

| Control | Description |
|---|---|
| **Ticker** | Any yfinance-compatible crypto ticker (BTC-USD, ETH-USD, etc.) or custom |
| **Period** | Lookback window for data fetch (7d to 90d) |
| **Interval** | Bar size — 1h, 2h, 4h, or 1d |
| **Gravity Lines** | Number of KMeans clusters (key levels to detect), 4–12 |
| **Level Lookback** | How many bars back to use for level detection |
| **Vol Window** | Rolling window for volatility calculation |
| **Display Bars** | How many recent bars to show on the chart |
| **SCAN NOW** | Clears cache and re-fetches fresh data |
| **Auto-refresh** | Clears cache on every rerun (every 5 min in Streamlit's rerun loop) |

---

## How the Setup Score Works

```
Setup Score = Compression Score (max 40) + Proximity Score (max 30) + ML Score (max 30)

Compression Score = max(0, (0.30 - vol_percentile) / 0.30) × 40
   → Full 40 pts when vol is at or below the 30th percentile

Proximity Score   = max(0, (0.03 - dist_to_nearest_pct) / 0.03) × 30
   → Full 30 pts when price is within 0–3% of a gravity line

ML Score          = breakout_probability × 30
   → Full 30 pts when the GBM model gives 100% breakout probability
```

Score interpretation:
- **70–100** → Prime setup. Compression, gravity alignment, and ML all agree.
- **45–69** → Developing. Watch for a trigger.
- **30–44** → Early stage. Insufficient confluence.
- **0–29** → No setup. Vol expanded or price is far from key levels.

---

## How the ML Model Works

The Gradient Boosted Classifier trains on the most recent 80% of bars and predicts whether volatility will increase by more than 50% within the next 5 bars.

**Features (blended from both source repos):**

From Volatility-Explosion-Tracker:
- `vol_pct` — rolling vol percentile
- `vol_z` — vol z-score vs 150-bar mean
- `vol_trend` — 5-bar vol change
- `vol_accel` — acceleration of vol trend
- `atr_pct` — ATR as a % of price
- `atr_trend` — 5-bar ATR change

From Crypto-Gravity-Lines:
- `dist_nearest` — distance to nearest gravity line as % of price
- `bb_width` — Bollinger Band width (secondary compression metric)

Shared:
- `price_mom` — 5-bar price momentum

If the model can't train (insufficient data or class imbalance), it falls back to a neutral 50% probability.

---

## Notes

- **No API key required.** Data is fetched via `yfinance` (Yahoo Finance).
- **Mock data fallback.** If yfinance fails (rate limits, network issues), the app generates synthetic BTC-like price data so the dashboard remains functional for exploration.
- **Not financial advice.** This is a research and visualization tool. Past volatility patterns do not guarantee future breakouts.

---

## License

MIT — fork it, extend it, build on it.
