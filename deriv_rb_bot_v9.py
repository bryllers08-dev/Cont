"""
+=========================================================================+
|  DERIV RANGE BOT  v9  —  RDBEAR  (ContainmentEstimator + SignalEngine)|
|                                                                         |
|  Stripped of all LSTM / LightGBM / training infrastructure.            |
|  The sole intelligence is the statistical layer:                        |
|                                                                         |
|  1. ContainmentEstimator                                                |
|       P = Φ((log(U/S)−μt)/σ√t) − Φ((log(L/S)−μt)/σ√t)               |
|       σ  = EWMA vol  (α=0.06, λ=0.94 — responds fast to vol spikes)   |
|       μ  = drift, 3σ-significance filtered (noise → zero)              |
|       H  = Hurst scaling (mean-reverting → boost, trending → reduce)   |
|                                                                         |
|  2. VolatilityRegime  — ATR-based; skip HIGH vol windows               |
|                                                                         |
|  3. Signal Persistence  — p_stat must clear the floor on 3             |
|       consecutive ticks before a trade fires (no single-tick spikes)   |
|                                                                         |
|  4. EV Gate  — p_stat × payout − (1−p_stat) > 0 (real payout)        |
|                                                                         |
|  5. AdaptiveThreshold  — tightens slightly on losses, relaxes on wins  |
|                                                                         |
|  6. SPRTMonitor  — passive edge tracker, warns if edge disappears      |
|                                                                         |
|  7. KellyStaker  — tiered stake: 35%→12%→7%→5% as balance grows      |
|       Hard cap 10% / $5 max. Min $0.35. Calibrated for $1 account      |
|                                                                         |
|  Start: collect 30 min of ticks → calibrate barriers → live trade      |
|                                                                         |
|  Requirements: pip install numpy scipy websocket-client pandas arch     |
|  Env:          DERIV_API_TOKEN                                          |
+=========================================================================+
"""

import csv, json, logging, math, os, sys, threading, time
import io
from collections import deque
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats
import websocket
import pandas as pd
try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False
    log_stub = logging.getLogger("DerivRB_v9")
    log_stub.warning("[SignalEngine] `arch` library not found — GARCH filter disabled. "
                     "Install with: pip install arch")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_sh = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_fh = logging.FileHandler("deriv_rb_bot_v9.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_sh, _fh])
log = logging.getLogger("DerivRB_v9")

DATA_DIR = os.environ.get("DATA_DIR", "rb_bot_data")
os.makedirs(DATA_DIR, exist_ok=True)


# ===========================================================================
# CONFIGURATION
# ===========================================================================
CONFIG = {
    # -- Deriv -----------------------------------------------------------------
    "app_id"    : 1089,
    "api_token" : os.environ.get("DERIV_API_TOKEN", ""),

    # -- Symbol ----------------------------------------------------------------
    "symbol"    : "RDBEAR",

    # -- Tick collection -------------------------------------------------------
    "collect_hours" : 0.5,          # 30 min history → plenty for estimator
    "data_dir"      : os.path.join(DATA_DIR, "tick_data"),
    "min_ticks"     : 500,          # minimum ticks before live trading starts

    # -- Expiry choices (minutes) ----------------------------------------------
    "hold_durations" : [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],  # extended to 15m

    # -- Barrier ---------------------------------------------------------------
    "expiryrange_barrier" : 2.97,   # fallback for short durations; long durs scale up
    "currency"            : "USD",

    # -- ContainmentEstimator --------------------------------------------------
    # Windows are in ticks. RDBEAR ticks ~180-240/min, so these are auto-scaled
    # by tpm at runtime — see ContainmentEstimator.from_tick_buf.
    # Values below are the MINUTES of data wanted; scaling happens in code.
    "containment_vol_mins"     : 2,     # 2 min of ticks for EWMA σ
    "containment_drift_mins"   : 1,     # 1 min of ticks for drift μ
    "containment_hurst_mins"   : 6,     # 6 min of ticks for Hurst R/S (covers longer durations)
    # Legacy tick-based fallbacks (used if tpm unavailable)
    "containment_vol_window"   : 60,
    "containment_drift_window" : 30,
    "containment_hurst_window" : 120,
    "containment_ewma_alpha"   : 0.06,  # EWMA decay (RiskMetrics λ=0.94)
    "containment_use_drift"    : True,
    "containment_use_hurst"    : True,
    "containment_hurst_scale"  : 0.5,
    "ticks_per_minute"         : None,  # None = auto-detect

    # -- Signal engine filters ------------------------------------------------
    "signal_engine_enabled"    : True,   # master switch for all three filters
    "se_log_interval"          : 30,     # ticks between SKIP log lines for SE

    # -- EV gate ---------------------------------------------------------------
    "ev_confidence_floor"  : 0.676,  # break-even at 48% payout = 1/1.48
    "min_ev_threshold"     : 0.05,   # raised from 0.001 — filters thin-edge trades
    "min_payout_pct"       : 0.48,   # reject proposals below 48%
    "ev_check_on_proposal" : True,

    # -- Signal persistence ----------------------------------------------------
    # p_stat must clear ev_confidence_floor on this many consecutive ticks
    # before a proposal is sent. Prevents single-tick spikes.
    "signal_persistence_ticks" : 3,

    # -- Volatility regime gate ------------------------------------------------
    "vol_regime_low_thr"   : 0.0008,
    "vol_regime_high_thr"  : 0.0030,
    "skip_high_vol_regime" : True,

    # -- Confidence threshold --------------------------------------------------
    "conf_floor"     : 0.50,
    "conf_ceil"      : 0.55,
    "conf_base"      : 0.50,
    "conf_loss_step" : 0.002,
    "conf_win_step"  : 0.002,

    # -- Kelly staking ---------------------------------------------------------
    # Calibrated for $1 starting balance.
    # stake_pct=0.35 → $0.35 on $1.00 (hits min), $0.70 on $2.00, scales up.
    # kelly_max_pct=0.50 prevents the 15% cap from cutting below min_stake.
    "kelly_fraction"  : 0.25,
    "kelly_max_pct"   : 0.10,    # hard cap: never more than 10% of balance per trade
    "kelly_min_stake" : 0.35,    # Deriv minimum
    "kelly_max_stake" : 5.0,     # absolute max stake regardless of balance
    "stake_pct"       : 0.35,    # used only as floor logic — see tiered_stake_pct below
    # Tiered stake % — high enough to hit $0.35 min at $1, tapers off quickly
    # balance $1.00-$1.99 → 35%  ($0.35–$0.70)
    # balance $2.00-$4.99 → 15%  ($0.30 → floored to $0.35 min .. $0.75)
    # balance $5.00+      →  7%  (conservative flat growth)

    # -- Risk limits -----------------------------------------------------------
    # Widened for $1 account — tight % limits halt the bot after a single loss.
    "max_daily_loss_pct"        : 0.80,   # stop only if balance drops to $0.20
    "take_profit_pct"           : 9999.0, # effectively unlimited
    "max_drawdown_from_peak_pct": 0.80,   # stop only on near-total drawdown

    # -- Consecutive-loss cooldown ---------------------------------------------
    # Bot never halts on losses — only brief cooldown, then resumes.
    "max_consec_losses"          : 5,
    "consec_loss_cooldown_ticks" : 60,    # short cooldown (≈1 min), then keep going

    # -- Post-trade cooldown ---------------------------------------------------
    "min_ticks_between_trades" : 60,

    # -- SPRT ------------------------------------------------------------------
    "sprt_p0"    : 0.50,
    "sprt_p1"    : 0.53,
    "sprt_alpha" : 0.10,
    "sprt_beta"  : 0.20,

    # -- Trade log -------------------------------------------------------------
    "trade_log" : os.path.join(DATA_DIR, "trade_log.csv"),
}

os.makedirs(CONFIG["data_dir"], exist_ok=True)


# ===========================================================================
# UTILITIES
# ===========================================================================

def wilson_ci(wins, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p   = wins / n
    denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ===========================================================================
# SPRT MONITOR
# ===========================================================================

class SPRTMonitor:
    """Sequential Probability Ratio Test — passive edge tracker."""
    def __init__(self, p0=0.50, p1=0.53, alpha=0.10, beta=0.20):
        self.A   = math.log((1-beta)/alpha)
        self.B   = math.log(beta/(1-alpha))
        self.p0  = p0; self.p1 = p1
        self.llr = 0.0; self.n = 0; self.wins = 0
        self.status = "CONTINUE"

    def update(self, win: bool) -> str:
        self.n += 1
        if win:
            self.wins += 1
            self.llr  += math.log(self.p1/self.p0)
        else:
            self.llr  += math.log((1-self.p1)/(1-self.p0))
        if   self.llr >= self.A: self.status = "ACCEPT_H1"; self.llr = 0.0
        elif self.llr <= self.B: self.status = "ACCEPT_H0"; self.llr = 0.0
        else:                    self.status = "CONTINUE"
        return self.status

    def summary(self):
        wr     = self.wins/self.n if self.n else 0.0
        lo, hi = wilson_ci(self.wins, self.n)
        return f"{self.status}  n={self.n}  WR={wr:.3f}  CI=[{lo:.3f},{hi:.3f}]"


# ===========================================================================
# CONTAINMENT ESTIMATOR  (drift + EWMA vol + Hurst)
# ===========================================================================

class ContainmentEstimator:
    """
    P(final tick within barrier) under drift-adjusted GBM with:
      - EWMA volatility  (α=0.06, responds fast to vol spikes)
      - 3σ-filtered drift (suppress noise-driven drift)
      - Hurst scaling with small-sample bias correction
    """

    @staticmethod
    def p_containment(price, upper, lower, sigma, t_ticks, mu=0.0):
        if sigma <= 1e-10 or t_ticks <= 0:
            return 0.99 if lower <= price <= upper else 0.01
        sd        = sigma * math.sqrt(t_ticks)
        log_price = math.log(max(price, 1e-8))
        log_half  = math.log(max(upper, 1e-8)) - log_price
        dt        = float(np.clip(mu * t_ticks,
                                  -0.35*abs(log_half),
                                   0.35*abs(log_half)))
        p_hi = scipy_stats.norm.cdf((math.log(max(upper, 1e-8)) - log_price - dt) / sd)
        p_lo = scipy_stats.norm.cdf((math.log(max(lower, 1e-8)) - log_price - dt) / sd)
        return float(np.clip(p_hi - p_lo, 0.01, 0.99))

    @staticmethod
    def _ewma_sigma(tick_buf, vol_window=60, alpha=0.06):
        buf = list(tick_buf)[-(vol_window+1):]
        if len(buf) < 3: return 0.001
        prices = np.array([t["price"] for t in buf], dtype=float)
        lr     = np.diff(np.log(np.maximum(prices, 1e-8)))
        if len(lr) < 2: return max(float(np.std(lr)), 1e-8)
        ewma_var = float(lr[0]**2)
        for r in lr[1:]:
            ewma_var = alpha*float(r**2) + (1.0-alpha)*ewma_var
        return max(math.sqrt(ewma_var), 1e-8)

    @staticmethod
    def _drift(tick_buf, drift_window=30, sigma=None):
        buf = list(tick_buf)[-(drift_window+1):]
        if len(buf) < 3: return 0.0
        prices = np.array([t["price"] for t in buf], dtype=float)
        lr     = np.diff(np.log(np.maximum(prices, 1e-8)))
        mu     = float(np.mean(lr))
        s      = sigma if sigma else max(float(np.std(lr)), 1e-8)
        std_err = s / math.sqrt(max(len(lr), 1))
        return mu if abs(mu) >= 3.0*std_err else 0.0

    @staticmethod
    def _hurst(tick_buf, hurst_window=120):
        buf = list(tick_buf)[-hurst_window:]
        if len(buf) < 20: return 0.5
        prices = np.array([t["price"] for t in buf], dtype=float)
        lr     = np.diff(np.log(np.maximum(prices, 1e-8)))
        n      = len(lr)
        if n < 4: return 0.5
        dev    = lr - np.mean(lr)
        cumdev = np.cumsum(dev)
        R      = float(np.max(cumdev) - np.min(cumdev))
        S      = float(np.std(lr))
        if S < 1e-10 or R <= 0: return 0.5
        H_raw = math.log(R/S) / math.log(n)
        return float(np.clip(H_raw - 0.45/math.sqrt(n), 0.1, 0.9))

    @staticmethod
    def _tpm(tick_buf, window=30):
        buf = list(tick_buf)[-window:]
        if len(buf) < 2: return 60.0
        dt = buf[-1]["timestamp"] - buf[0]["timestamp"]
        return (len(buf)-1)/dt*60.0 if dt > 0 else 60.0

    @classmethod
    def from_tick_buf(cls, tick_buf, barrier_offset, duration_mins,
                      vol_window=60, ticks_per_minute=None,
                      drift_window=30, hurst_window=120,
                      ewma_alpha=0.06, use_drift=True, use_hurst=True,
                      hurst_scale=0.5,
                      vol_mins=None, drift_mins=None, hurst_mins=None):
        if not tick_buf: return 0.5
        price  = list(tick_buf)[-1]["price"]
        upper  = price + barrier_offset
        lower  = price - barrier_offset
        tpm    = ticks_per_minute or cls._tpm(tick_buf)
        # Auto-scale windows by tpm so they represent real time, not raw tick count.
        # vol_mins/drift_mins/hurst_mins take priority over legacy tick counts.
        v_win  = max(30, int(tpm * vol_mins))   if vol_mins   else vol_window
        d_win  = max(15, int(tpm * drift_mins)) if drift_mins else drift_window
        h_win  = max(60, int(tpm * hurst_mins)) if hurst_mins else hurst_window
        sigma  = cls._ewma_sigma(tick_buf, v_win, ewma_alpha)
        t_tick = max(1.0, tpm * duration_mins)
        mu     = cls._drift(tick_buf, d_win, sigma) if use_drift else 0.0
        p      = cls.p_containment(price, upper, lower, sigma, t_tick, mu)
        if use_hurst:
            H      = cls._hurst(tick_buf, h_win)
            factor = float(np.clip(1.0 + hurst_scale*(0.5-H), 0.75, 1.25))
            p      = float(np.clip(p * factor, 0.01, 0.99))
        return p

    @staticmethod
    def ev(p_win, payout_ratio):
        return float(p_win * payout_ratio - (1.0 - p_win))


# ===========================================================================
# VOLATILITY REGIME
# ===========================================================================

class VolatilityRegime:
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"

    @staticmethod
    def from_tick_buf(tick_buf, window=20, low_thr=0.0008, high_thr=0.0030):
        buf = list(tick_buf)[-(window+1):]
        if len(buf) < 3: return VolatilityRegime.MEDIUM
        prices   = np.array([t["price"] for t in buf], dtype=float)
        last     = max(prices[-1], 1e-8)
        atr_norm = float(np.std(np.diff(prices)) / last)
        if atr_norm < low_thr:  return VolatilityRegime.LOW
        if atr_norm >= high_thr: return VolatilityRegime.HIGH
        return VolatilityRegime.MEDIUM


# ===========================================================================
# ADAPTIVE THRESHOLD
# ===========================================================================

class AdaptiveThreshold:
    def __init__(self, cfg):
        self.base      = self.threshold = cfg["conf_base"]
        self.floor     = cfg["conf_floor"]
        self.ceil      = cfg["conf_ceil"]
        self.ls        = cfg["conf_loss_step"]
        self.ws        = cfg["conf_win_step"]

    def update(self, win: bool):
        if win:
            self.threshold = max(self.floor, self.threshold - self.ws)
        else:
            self.threshold = min(self.ceil,  self.threshold + self.ls)

    def get(self) -> float:
        return self.threshold

    def reset(self):
        self.threshold = self.base


# ===========================================================================
# KELLY STAKER
# ===========================================================================

class KellyStaker:
    """
    Tiered stake sizing calibrated for $1 account:
      $1.00–$1.99  → 35% (ensures $0.35 Deriv minimum is met)
      $2.00–$4.99  → 12% (tapering off)
      $5.00–$14.99 →  7% (conservative growth)
      $15.00+      →  5% (steady compounding)
    Hard cap: 10% of balance per trade, $5.00 absolute maximum.
    """
    def __init__(self, cfg):
        self.fraction  = cfg["kelly_fraction"]
        self.max_pct   = cfg["kelly_max_pct"]
        self.min_stake = cfg["kelly_min_stake"]
        self.max_stake = cfg["kelly_max_stake"]
        self.stake_pct = cfg.get("stake_pct", 0.035)
        self.wins = self.n = 0

    def _tiered_pct(self, balance):
        """Step down stake % as balance grows — hits min at $1, conservative above."""
        if balance < 2.00:  return 0.35   # $1.00–$1.99 → 35%  (floor needed for $0.35 min)
        if balance < 5.00:  return 0.12   # $2.00–$4.99 → 12%  ($0.24–$0.60, floored to $0.35)
        if balance < 15.00: return 0.07   # $5.00–$14.99→  7%  ($0.35–$1.05)
        return 0.05                        # $15.00+     →  5%  (steady growth)

    def next_stake(self, p_win, balance, payout_ratio=0.48):
        if balance <= 0: return self.min_stake
        # Tiered proportional stake — scales down as balance grows
        prop_stake  = balance * self._tiered_pct(balance)
        # Kelly component (quarter-Kelly)
        b           = payout_ratio; q = 1.0 - p_win
        f_star      = (p_win*b - q) / b
        kelly_stake = min(f_star*self.fraction, self.max_pct)*balance if f_star > 0 else 0.0
        # Take the larger of tiered prop vs Kelly, then hard-cap at 10% and max_stake
        stake = max(prop_stake, kelly_stake)
        stake = min(stake, self.max_stake, balance * self.max_pct)
        # Always honour Deriv minimum
        stake = max(stake, self.min_stake)
        # Never stake more than balance
        stake = min(stake, balance)
        return round(stake, 2)

    def record(self, win: bool):
        if win: self.wins += 1
        self.n += 1
        log.info("[Kelly] %s  WR=%.1f%%  stake_next_based_on_bal",
                 "WIN" if win else "LOSS",
                 self.wins/self.n*100 if self.n else 0)


# ===========================================================================
# TRADE LOGGER  (CSV)
# ===========================================================================

class TradeLogger:
    FIELDS = [
        "timestamp", "symbol", "duration_mins",
        "barrier_offset", "price_at_entry",
        "p_stat", "ev", "payout",
        "sigma_ewma", "mu_drift", "hurst",
        "vol_regime", "adaptive_threshold",
        "stake", "balance_before", "outcome", "profit", "balance_after",
        "sprt_status", "session_wr",
    ]

    def __init__(self, path):
        self.path    = path
        self._exists = os.path.isfile(path)

    def log(self, row: dict):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
            if not self._exists:
                w.writeheader()
                self._exists = True
            w.writerow(row)


# ===========================================================================
# HISTORICAL COLLECTOR
# ===========================================================================

class HistoricalCollector:
    WS_URL       = "wss://ws.binaryws.com/websockets/v3"
    MAX_PER_CALL = 5000

    def __init__(self, symbol, cfg, done, existing_df=None):
        self.symbol    = symbol
        self.cfg       = cfg
        self.done      = done
        self._existing = existing_df

    def _fetch_page(self, end_epoch):
        import queue as _q
        q = _q.Queue()

        def _on_open(ws):
            ws.send(json.dumps({
                "ticks_history": self.symbol, "end": end_epoch,
                "count": self.MAX_PER_CALL, "style": "ticks",
                "adjust_start_time": 1,
            }))

        def _on_msg(ws, raw):
            try:
                msg = json.loads(raw)
                if msg.get("msg_type") == "history":
                    h = msg.get("history", {})
                    q.put(sorted(
                        [{"timestamp": float(t), "price": float(p)}
                         for t, p in zip(h.get("times",[]), h.get("prices",[]))],
                        key=lambda x: x["timestamp"]
                    ))
                elif "error" in msg:
                    log.error("[Collector] Error: %s", msg["error"])
                    q.put([])
                else:
                    return
                ws.close()
            except Exception as e:
                log.warning("[Collector] Parse error: %s", e)
                q.put([]); ws.close()

        def _on_err(ws, e):
            log.warning("[Collector] WS error: %s", e); q.put([])

        ws = websocket.WebSocketApp(
            f"{self.WS_URL}?app_id={self.cfg['app_id']}",
            on_open=_on_open, on_message=_on_msg,
            on_error=_on_err, on_close=lambda *_: None,
        )
        t = threading.Thread(target=ws.run_forever, daemon=True)
        t.start()
        try:
            return q.get(timeout=25)
        except Exception:
            return []
        finally:
            try: ws.close()
            except Exception: pass
            t.join(timeout=3)

    def _collect(self):
        cfg          = self.cfg
        collect_secs = int(cfg["collect_hours"] * 3600)
        now_epoch    = int(time.time())
        cutoff_epoch = now_epoch - collect_secs

        log.info("[Collector/%s] Fetching %.1fh of ticks ...",
                 self.symbol, cfg["collect_hours"])

        all_ticks = []
        if self._existing is not None and not self._existing.empty:
            in_win = self._existing[self._existing["timestamp"] >= cutoff_epoch]
            if not in_win.empty:
                all_ticks = in_win.to_dict("records")
                if min(t["timestamp"] for t in all_ticks) <= cutoff_epoch:
                    log.info("[Collector/%s] Existing CSV covers window — reusing.",
                             self.symbol)
                    self._save(all_ticks); return

        fetch_end = now_epoch
        for page_num in range(1, 20):
            page = self._fetch_page(fetch_end)
            if not page:
                log.warning("[Collector/%s] Empty page %d.", self.symbol, page_num)
                break
            all_ticks.extend([tk for tk in page if tk["timestamp"] >= cutoff_epoch])
            if page[0]["timestamp"] <= cutoff_epoch:
                break
            fetch_end = int(page[0]["timestamp"]) - 1
            time.sleep(0.3)

        log.info("[Collector/%s] Collected %d ticks.", self.symbol, len(all_ticks))
        self._save(all_ticks)

    def _save(self, ticks):
        os.makedirs(self.cfg["data_dir"], exist_ok=True)
        path = os.path.join(self.cfg["data_dir"], f"ticks_{self.symbol}.csv")
        if ticks:
            df = (pd.DataFrame(ticks)
                  .drop_duplicates(subset=["timestamp"])
                  .sort_values("timestamp")
                  .reset_index(drop=True))
            df.to_csv(path, index=False)
            log.info("[Collector/%s] Saved %d ticks → %s",
                     self.symbol, len(df), path)
        self.done.set()

    def start(self):
        def _run():
            try:
                self._collect()
            except Exception as e:
                log.error("[Collector/%s] Fatal: %s", self.symbol, e, exc_info=True)
                self.done.set()
        threading.Thread(target=_run, daemon=True,
                         name=f"HistCol-{self.symbol}").start()


# ===========================================================================
# LIVE TRADER
# ===========================================================================

class LiveTrader:
    WS_URL = "wss://ws.binaryws.com/websockets/v3"

    def __init__(self, cfg, initial_ticks, staker, threshold, sprt, trade_logger):
        self.cfg          = cfg
        self.staker       = staker
        self.threshold    = threshold
        self.sprt         = sprt
        self.logger       = trade_logger

        # Tick buffer — seeded with historical ticks, grows with live ticks
        self.tick_buf     = deque(initial_ticks, maxlen=5000)

        # Per-duration calibrated barriers and payout cache
        self.barrier_table = {}   # {duration_mins: barrier_offset}
        self.payout_table  = {}   # {duration_mins: payout_ratio}

        # State
        self.ws             = None
        self.balance        = None
        self.start_balance  = None
        self.peak_balance   = None
        self.running        = False

        # Trade state
        self.waiting_result   = False
        self.waiting_proposal = False
        self.pending_pid      = None
        self.pending_stake    = 0.0
        self.pending_dur      = None
        self.pending_p_stat   = 0.0
        self.pending_barrier  = 0.0
        self.pending_payout   = 0.0

        # Counters
        self.live_tick_count   = 0
        self.post_trade_tick   = 0
        self.consec_losses     = 0
        self.cooldown_until    = 0
        self.signal_engine     = SignalEngine()
        self.wins = self.total = 0
        self.session_pnl       = 0.0

        # Signal persistence counter
        self._persist_count    = 0   # consecutive ticks above floor
        self._persist_best_p   = 0.0 # best p_stat in persistence window
        self._persist_dur      = None

        # Cache last computed intelligence values (for logging/CSV)
        self._last_p_stat      = 0.0
        self._last_ev          = 0.0
        self._last_sigma       = 0.0
        self._last_mu          = 0.0
        self._last_hurst       = 0.5
        self._last_regime      = VolatilityRegime.MEDIUM

        # Barrier calibration flag
        self._calibrating      = False

    # ── WebSocket lifecycle ───────────────────────────────────────────────

    def run(self):
        self.running = True
        log.info("[Trader] Connecting ...")
        self.ws = websocket.WebSocketApp(
            f"{self.WS_URL}?app_id={self.cfg['app_id']}",
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
        )
        while self.running:
            try:
                self.ws.run_forever(ping_interval=25, ping_timeout=15)
            except Exception as e:
                log.error("[Trader] WS error: %s", e)
            if self.running:
                log.info("[Trader] Reconnecting in 3s ...")
                time.sleep(3)

    def stop(self):
        self.running = False
        try: self.ws.close()
        except Exception: pass

    def _on_open(self, ws):
        log.info("[Trader] Connected. Authorising ...")
        ws.send(json.dumps({
            "authorize": self.cfg["api_token"],
        }))

    def _on_error(self, ws, error):
        log.warning("[Trader] WS error: %s", error)

    def _on_close(self, ws, code, msg):
        log.info("[Trader] WS closed (%s %s)", code, msg)

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        mt = msg.get("msg_type", "")
        try:
            if   mt == "authorize":      self._on_auth(msg)
            elif mt == "balance":        self._on_balance(msg)
            elif mt == "tick":           self._on_tick(msg)
            elif mt == "proposal":       self._on_proposal(msg)
            elif mt == "buy":            self._on_buy(msg)
            elif mt == "proposal_open_contract": self._on_poc(msg)
            elif "error" in msg:
                log.warning("[Trader] API error: %s",
                            msg["error"].get("message", str(msg["error"])))
        except Exception as e:
            log.error("[Trader] Message handler error: %s", e, exc_info=True)

    # ── Auth ─────────────────────────────────────────────────────────────

    def _on_auth(self, msg):
        info = msg.get("authorize", {})
        self.balance       = float(info.get("balance", 0))
        self.start_balance = self.balance
        self.peak_balance  = self.balance
        log.info("[Trader] Authorised: %s | %s %.2f",
                 info.get("loginid", "?"),
                 info.get("currency", "USD"),
                 self.balance)
        # Subscribe to balance updates
        self.ws.send(json.dumps({"balance": 1, "subscribe": 1}))
        # Subscribe to live ticks
        self.ws.send(json.dumps({
            "ticks": self.cfg["symbol"], "subscribe": 1,
        }))
        # Launch barrier calibration in background
        threading.Thread(target=self._calibrate_barriers,
                         daemon=True, name="BarrierCal").start()

        stake_preview = round(self.balance * self.cfg.get("stake_pct", 0.035), 2)
        log.info("[Staker] %.1f%% of balance → first stake ~$%.2f",
                 self.cfg.get("stake_pct",0.035)*100, stake_preview)
        log.info("[Trader] Waiting for tick buffer to fill "
                 "(need %d ticks) ...", self.cfg["min_ticks"])

    # ── Balance ───────────────────────────────────────────────────────────

    def _on_balance(self, msg):
        b = msg.get("balance", {})
        if "balance" in b:
            self.balance = float(b["balance"])
            if self.peak_balance is None or self.balance > self.peak_balance:
                self.peak_balance = self.balance

    # ── Tick ──────────────────────────────────────────────────────────────

    def _on_tick(self, msg):
        t = msg.get("tick", {})
        self.tick_buf.append({
            "timestamp": float(t.get("epoch", time.time())),
            "price"    : float(t.get("quote", 0)),
        })
        self.live_tick_count += 1

        if not self.running or self.waiting_result or self.waiting_proposal:
            return

        # Not enough ticks yet
        if len(self.tick_buf) < self.cfg["min_ticks"]:
            if self.live_tick_count % 100 == 0:
                log.info("[Trader] Buffer %d/%d ticks ...",
                         len(self.tick_buf), self.cfg["min_ticks"])
            return

        # Cooldown
        if self.live_tick_count < self.cooldown_until:
            return

        # Post-trade cooldown
        min_gap = self.cfg.get("min_ticks_between_trades", 60)
        if self.live_tick_count - self.post_trade_tick < min_gap:
            return

        self._evaluate_signal()

    # ── Signal evaluation ─────────────────────────────────────────────────

    def _evaluate_signal(self):
        cfg      = self.cfg
        ev_floor = cfg.get("ev_confidence_floor", 0.676)

        # ── 1. Volatility regime ───────────────────────────────────────────
        regime = VolatilityRegime.from_tick_buf(
            self.tick_buf,
            low_thr  = cfg.get("vol_regime_low_thr",  0.0008),
            high_thr = cfg.get("vol_regime_high_thr", 0.0030),
        )
        self._last_regime = regime
        if cfg.get("skip_high_vol_regime", True) and regime == VolatilityRegime.HIGH:
            self._persist_count = 0
            return

        # ── 1b. Signal Engine — three pre-entry filters ──────────────────────
        if cfg.get("signal_engine_enabled", True):
            se_pass, se_details = self.signal_engine.all_pass(
                self.tick_buf, self.live_tick_count)
            if not se_pass:
                self._persist_count = 0
                if self.live_tick_count % cfg.get("se_log_interval", 30) == 0:
                    log.info("[SE] SKIP — filter=%s H=%.4f garch_pct=%.1f gbm_prob=%.4f",
                             se_details.get("failed", "?"),
                             se_details.get("H", 0),
                             se_details.get("garch_pct", 0),
                             se_details.get("gbm_prob", 0))
                return

        # ── 2. ContainmentEstimator across all durations — pick best ──────
        fallback  = cfg.get("expiryrange_barrier", 2.97)
        durations = cfg.get("hold_durations", [2, 3, 4, 5, 6])

        best_p    = 0.0
        best_dur  = None

        for dur in durations:
            barrier = self.barrier_table.get(dur, fallback)
            p = ContainmentEstimator.from_tick_buf(
                self.tick_buf,
                barrier_offset   = barrier,
                duration_mins    = dur,
                vol_window       = cfg.get("containment_vol_window",    60),
                ticks_per_minute = cfg.get("ticks_per_minute",          None),
                drift_window     = cfg.get("containment_drift_window",  30),
                hurst_window     = cfg.get("containment_hurst_window",  120),
                ewma_alpha       = cfg.get("containment_ewma_alpha",    0.06),
                use_drift        = cfg.get("containment_use_drift",     True),
                use_hurst        = cfg.get("containment_use_hurst",     True),
                hurst_scale      = cfg.get("containment_hurst_scale",   0.5),
                vol_mins         = cfg.get("containment_vol_mins",      None),
                drift_mins       = cfg.get("containment_drift_mins",    None),
                hurst_mins       = cfg.get("containment_hurst_mins",    None),
            )
            if p > best_p:
                best_p   = p
                best_dur = dur

        # Cache for logging
        self._last_p_stat = best_p

        # Also cache internals for CSV
        buf = self.tick_buf
        self._last_sigma = ContainmentEstimator._ewma_sigma(buf)
        self._last_mu    = ContainmentEstimator._drift(buf, sigma=self._last_sigma)
        self._last_hurst = ContainmentEstimator._hurst(buf)

        # ── 3. EV gate (prelim — uses cached payout) ──────────────────────
        cached_payout = self.payout_table.get(best_dur,
                        cfg.get("min_payout_pct", 0.48))
        ev = ContainmentEstimator.ev(best_p, cached_payout)

        if best_p < ev_floor or ev < cfg.get("min_ev_threshold", 0.05):
            self._persist_count = 0
            if self.live_tick_count % 30 == 0:
                log.info("[Intel] p_stat=%.4f ev=%+.4f regime=%s  SKIP",
                         best_p, ev, regime)
            return

        # ── 4. AdaptiveThreshold ──────────────────────────────────────────
        if best_p < self.threshold.get():
            self._persist_count = 0
            return

        # ── 5. Signal persistence ──────────────────────────────────────────
        required = cfg.get("signal_persistence_ticks", 3)
        if best_dur == self._persist_dur:
            self._persist_count += 1
            if best_p > self._persist_best_p:
                self._persist_best_p = best_p
        else:
            # Duration changed — restart count on the new duration
            self._persist_count  = 1
            self._persist_best_p = best_p
            self._persist_dur    = best_dur

        if self._persist_count < required:
            if self.live_tick_count % 10 == 0:
                log.info("[Intel] Persistence %d/%d  p=%.4f  dur=%dm",
                         self._persist_count, required, best_p, best_dur)
            return

        # ── All checks passed — send proposal ─────────────────────────────
        se_info = ""
        if cfg.get("signal_engine_enabled", True):
            _, _se = self.signal_engine.all_pass(self.tick_buf, self.live_tick_count)
            se_info = (f" H={_se.get('H',0):.3f}"
                       f" garch={_se.get('garch_pct',0):.0f}pct"
                       f" gbm={_se.get('gbm_prob',0):.3f}")
        log.info("[Intel] *** SIGNAL *** p_stat=%.4f ev=%+.4f "
                 "dur=%dm regime=%s persistence=%d%s  → sending proposal",
                 best_p, ev, best_dur, regime, self._persist_count, se_info)

        self._persist_count = 0   # reset after firing

        # Compute stake now using cached payout
        if not self.balance or self.balance <= 0:
            log.warning("[Trader] Balance not confirmed — skipping.")
            return

        stake = self.staker.next_stake(best_p, self.balance, cached_payout)

        # Store pending trade info
        self.pending_stake    = stake
        self.pending_dur      = best_dur
        self.pending_p_stat   = best_p
        self.pending_barrier  = self.barrier_table.get(best_dur, fallback)
        self.pending_payout   = cached_payout
        self.waiting_proposal = True

        barrier_offset = self.barrier_table.get(best_dur, fallback)
        self.ws.send(json.dumps({
            "proposal"      : 1,
            "amount"        : stake,
            "basis"         : "stake",
            "contract_type" : "EXPIRYRANGE",
            "currency"      : cfg.get("currency", "USD"),
            "duration"      : best_dur,
            "duration_unit" : "m",
            "symbol"        : cfg["symbol"],
            "barrier"       : f"+{barrier_offset:.2f}",
            "barrier2"      : f"-{barrier_offset:.2f}",
        }))

    # ── Proposal response ─────────────────────────────────────────────────

    def _on_proposal(self, msg):
        if "error" in msg:
            log.warning("[Trader] Proposal error: %s",
                        msg["error"].get("message", str(msg["error"])))
            self.waiting_proposal = False
            return

        prop          = msg.get("proposal", {})
        pid           = msg.get("id") or prop.get("id")
        ask_price     = float(prop.get("ask_price",
                              prop.get("cost", self.pending_stake)))
        payout_amount = float(prop.get("payout", ask_price * 1.48))
        payout_ratio  = round((payout_amount - ask_price) / max(ask_price, 1e-8), 4)

        cfg      = self.cfg
        ev_floor = cfg.get("ev_confidence_floor", 0.676)
        ev_thr   = cfg.get("min_ev_threshold",    0.05)
        min_pay  = cfg.get("min_payout_pct",       0.48)

        # Payout minimum check
        if payout_ratio < min_pay:
            log.info("[EV] SKIP — payout=%.1f%% < min=%.0f%%",
                     payout_ratio*100, min_pay*100)
            self.waiting_proposal = False
            return

        # Final EV check with real payout
        p   = self.pending_p_stat
        ev  = ContainmentEstimator.ev(p, payout_ratio)
        self._last_ev = ev

        if cfg.get("ev_check_on_proposal", True):
            if p < ev_floor or ev < ev_thr:
                log.info("[EV] SKIP — p=%.4f < floor=%.3f or EV=%+.4f < thr",
                         p, ev_floor, ev)
                self.waiting_proposal = False
                return

        self.pending_pid    = pid
        self.pending_payout = payout_ratio
        self.waiting_proposal = False
        self.waiting_result   = True

        log.info("[Trader] Proposal OK  payout=%.1f%%  p=%.4f  "
                 "EV=%+.4f  stake=$%.2f  pid=%s",
                 payout_ratio*100, p, ev,
                 self.pending_stake, pid)

        log.info("[Trader] Buying EXPIRYRANGE  stake=$%.2f  pid=%s",
                 self.pending_stake, pid)

        self.ws.send(json.dumps({
            "buy"  : pid,
            "price": self.pending_stake,
        }))

    # ── Buy confirmation ──────────────────────────────────────────────────

    def _on_buy(self, msg):
        if "error" in msg:
            log.warning("[Trader] Buy error: %s",
                        msg["error"].get("message", str(msg["error"])))
            self.waiting_result = False
            return
        buy = msg.get("buy", {})
        cid = str(buy.get("contract_id", "?"))
        log.info("[Trader] Contract opened  cid=%s  paid=$%.2f",
                 cid, float(buy.get("buy_price", self.pending_stake)))
        # Subscribe to settlement
        self.ws.send(json.dumps({
            "proposal_open_contract": 1,
            "contract_id"           : int(cid),
            "subscribe"             : 1,
        }))

    # ── Settlement ────────────────────────────────────────────────────────

    def _on_poc(self, msg):
        poc = msg.get("proposal_open_contract", {})
        if not poc.get("is_sold", 0):
            return  # not settled yet

        profit     = float(poc.get("profit", 0))
        win        = profit > 0
        new_bal    = float(poc.get("balance_after", self.balance or 0))
        old_bal    = self.balance or new_bal
        self.balance = new_bal
        if self.peak_balance is None or new_bal > self.peak_balance:
            self.peak_balance = new_bal

        self.session_pnl += profit
        self.total += 1
        if win:
            self.wins += 1
            self.consec_losses = 0
        else:
            self.consec_losses += 1
        self.signal_engine.consec_losses = self.consec_losses

        wr = self.wins / self.total * 100

        log.info("[Trade] %s #%d | P&L=%+.2f | bal=$%.2f | "
                 "W/L=%d/%d (%.1f%%) | SPRT:%s",
                 "WIN " if win else "LOSS",
                 self.total, profit, new_bal,
                 self.wins, self.total-self.wins, wr,
                 self.sprt.update(win))

        # Update intelligence components
        self.threshold.update(win)
        self.staker.record(win)
        self.post_trade_tick = self.live_tick_count
        self.waiting_result  = False

        # Consecutive loss cooldown
        if (not win and
                self.consec_losses >= self.cfg.get("max_consec_losses", 3)):
            cd = self.cfg.get("consec_loss_cooldown_ticks", 120)
            self.cooldown_until = self.live_tick_count + cd
            log.warning("[Risk] %d consecutive losses — cooling down %d ticks.",
                        self.consec_losses, cd)
            self.consec_losses = 0

        # CSV log
        self.logger.log({
            "timestamp"          : datetime.utcnow().isoformat(),
            "symbol"             : self.cfg["symbol"],
            "duration_mins"      : self.pending_dur,
            "barrier_offset"     : round(self.pending_barrier, 4),
            "price_at_entry"     : round(list(self.tick_buf)[-1]["price"], 5)
                                   if self.tick_buf else "",
            "p_stat"             : round(self.pending_p_stat, 5),
            "ev"                 : round(self._last_ev, 5),
            "payout"             : round(self.pending_payout, 4),
            "sigma_ewma"         : round(self._last_sigma, 8),
            "mu_drift"           : round(self._last_mu, 8),
            "hurst"              : round(self._last_hurst, 4),
            "vol_regime"         : self._last_regime,
            "adaptive_threshold" : round(self.threshold.get(), 4),
            "stake"              : round(self.pending_stake, 2),
            "balance_before"     : round(old_bal, 2),
            "outcome"            : "WIN" if win else "LOSS",
            "profit"             : round(profit, 2),
            "balance_after"      : round(new_bal, 2),
            "sprt_status"        : self.sprt.status,
            "session_wr"         : round(wr, 2),
        })


        # Recalibrate barriers in background after every trade
        threading.Thread(target=self._calibrate_barriers,
                         daemon=True, name="BarrierRecal").start()

        # Risk checks
        if not self._risk_ok():
            log.warning("[Risk] Risk limit hit — stopping.")
            self.stop()

    # ── Risk ──────────────────────────────────────────────────────────────

    def _risk_ok(self):
        cfg = self.cfg
        sb  = self.start_balance or 1.0
        pb  = self.peak_balance  or self.balance or 1.0
        bl  = self.balance       or 0.0

        if self.session_pnl <= -(sb * cfg["max_daily_loss_pct"]):
            log.warning("[Risk] Daily loss limit: P&L=%.2f", self.session_pnl)
            return False
        if bl > 0 and (pb - bl) >= sb * cfg["max_drawdown_from_peak_pct"]:
            log.warning("[Risk] Drawdown limit: peak=%.2f current=%.2f", pb, bl)
            return False
        if self.session_pnl >= sb * cfg.get("take_profit_pct", 9999):
            log.info("[Risk] Take-profit reached: P&L=%.2f", self.session_pnl)
            return False
        return True

    # ── Barrier calibration ───────────────────────────────────────────────

    def _calibrate_barriers(self):
        if self._calibrating: return
        self._calibrating = True

        cfg       = self.cfg
        durations = cfg.get("hold_durations", [2, 3, 4, 5, 6])
        fallback  = cfg.get("expiryrange_barrier", 2.97)
        symbol    = cfg["symbol"]

        log.info("[BarrierCal] Calibrating %s for durations %s ...",
                 symbol, durations)

        import queue as _q

        def _probe(duration, barrier):
            """Single-shot proposal to get payout for this barrier+duration."""
            q = _q.Queue()

            def _oo(ws):
                ws.send(json.dumps({
                    "proposal"      : 1,
                    "amount"        : cfg["kelly_min_stake"],
                    "basis"         : "stake",
                    "contract_type" : "EXPIRYRANGE",
                    "currency"      : cfg["currency"],
                    "duration"      : duration,
                    "duration_unit" : "m",
                    "symbol"        : symbol,
                    "barrier"       : f"+{barrier:.2f}",
                    "barrier2"      : f"-{barrier:.2f}",
                }))

            def _om(ws, raw):
                try:
                    m = json.loads(raw)
                    if m.get("msg_type") == "proposal":
                        p = m.get("proposal", {})
                        ask = float(p.get("ask_price", cfg["kelly_min_stake"]))
                        pay = float(p.get("payout",   ask*1.48))
                        q.put((pay - ask) / max(ask, 1e-8))
                    elif "error" in m:
                        q.put(None)
                    else:
                        return
                    ws.close()
                except Exception:
                    q.put(None); ws.close()

            ws2 = websocket.WebSocketApp(
                f"{self.WS_URL}?app_id={cfg['app_id']}",
                on_open=_oo, on_message=_om,
                on_error=lambda *_: q.put(None),
                on_close=lambda *_: None,
            )
            t = threading.Thread(target=ws2.run_forever, daemon=True)
            t.start()
            try:
                return q.get(timeout=15)
            except Exception:
                return None
            finally:
                try: ws2.close()
                except Exception: pass
                t.join(timeout=3)

        # Per-duration search bounds (lo, hi, payout_lo, payout_hi).
        # Barriers grow with duration — longer hold needs wider range to contain.
        dur_bounds = {
            2 : (0.50,  5.0,  0.46, 0.50),
            3 : (0.80,  6.0,  0.46, 0.50),
            4 : (1.00,  7.0,  0.46, 0.50),
            5 : (1.20,  8.0,  0.46, 0.50),
            6 : (1.40,  9.0,  0.46, 0.50),
            7 : (1.60, 10.5,  0.46, 0.50),
            8 : (1.80, 12.0,  0.46, 0.50),
            9 : (2.00, 13.5,  0.46, 0.50),
            10: (2.20, 15.0,  0.46, 0.50),
            12: (2.60, 17.5,  0.46, 0.50),
            15: (3.00, 21.0,  0.46, 0.50),
        }
        for dur in durations:
            lo, hi, p_lo, p_hi = dur_bounds.get(dur, (0.5, 8.0, 0.46, 0.50))
            best = fallback; best_pr = None
            for _ in range(10):
                mid = round((lo + hi) / 2, 2)
                pr  = _probe(dur, mid)
                if pr is None: break
                if best_pr is None or abs(pr-0.485) < abs(best_pr-0.485):
                    best = mid; best_pr = pr
                if p_lo <= pr <= p_hi: break
                if pr < p_lo: hi = mid
                else:         lo = mid
                time.sleep(0.3)

            self.barrier_table[dur] = best
            if best_pr is not None:
                self.payout_table[dur] = float(best_pr)
            log.info("[BarrierCal] %s %dm → barrier=±%.2f  payout=%.1f%%",
                     symbol, dur, best,
                     (best_pr or 0)*100)

        log.info("[BarrierCal] Done: %s",
                 {f"{k}m": f"±{v:.2f}" for k, v in self.barrier_table.items()})
        self._calibrating = False



# ===========================================================================
# SIGNAL ENGINE  — Three-filter pre-entry gate
# ===========================================================================

class SignalEngine:
    """
    Three independent filters that must ALL pass before any trade entry.

    Filter 1 — Hurst Exponent (R/S analysis, 200-tick window)
        H < 0.48  → mean-reverting market  → PASS
        Recomputed every 50 ticks and cached.
        After 2+ consecutive losses: tightened to H < 0.44.

    Filter 2 — GARCH(1,1) volatility (arch library, 300 log-returns)
        Forecast one-step-ahead σ must be in the bottom 35th percentile
        of its own 500-step rolling history → entering only in calm periods.
        Recomputed every 30 ticks and cached.
        Falls back to EWMA vol comparison if `arch` is unavailable.

    Filter 3 — GBM barrier probability (analytic reflection principle)
        P(path stays within ±k·σ over 5 ticks) ≥ 0.62
        k is solved so the threshold is just met; we check the probability
        directly using the exact formula for Brownian motion first-passage.
    """

    # Rolling history lengths
    HURST_WINDOW     = 200
    HURST_RECOMPUTE  = 50    # ticks between Hurst recomputations
    GARCH_WINDOW     = 300   # log-returns for GARCH fit
    GARCH_RECOMPUTE  = 30    # ticks between GARCH recomputations
    GARCH_HIST_LEN   = 500   # rolling σ history for percentile calculation
    GBM_TICKS        = 5     # horizon for GBM stay-in-bounds check
    GBM_PROB_FLOOR   = 0.62  # minimum stay-in-bounds probability

    def __init__(self):
        # Hurst cache
        self._hurst_cache : float | None = None
        self._hurst_tick  : int          = -999

        # GARCH cache
        self._garch_sigma_cache : float | None = None
        self._garch_tick        : int           = -999
        self._garch_sigma_hist  : deque         = deque(maxlen=self.GARCH_HIST_LEN)

        # streak tracking (set from outside)
        self.consec_losses : int = 0

    # ── Hurst (R/S analysis) ──────────────────────────────────────────────

    @staticmethod
    def _compute_hurst(prices: np.ndarray) -> float:
        """Classic R/S Hurst exponent on a price series."""
        n = len(prices)
        if n < 20:
            return 0.5
        log_ret = np.diff(np.log(prices + 1e-12))
        # Split into sub-series of increasing length and compute R/S
        lags = []
        rs_vals = []
        for size in [max(10, n // 8), max(15, n // 4), max(20, n // 2), n]:
            seg = log_ret[:size]
            mean = seg.mean()
            dev  = np.cumsum(seg - mean)
            R    = dev.max() - dev.min()
            S    = seg.std(ddof=1)
            if S > 0:
                lags.append(np.log(size))
                rs_vals.append(np.log(R / S))
        if len(lags) < 2:
            return 0.5
        H, _ = np.polyfit(lags, rs_vals, 1)
        return float(np.clip(H, 0.0, 1.0))

    def hurst(self, tick_buf, current_tick: int) -> float:
        """Return cached Hurst, recomputing every HURST_RECOMPUTE ticks."""
        if (self._hurst_cache is None or
                current_tick - self._hurst_tick >= self.HURST_RECOMPUTE):
            buf = list(tick_buf)[-self.HURST_WINDOW:]
            if len(buf) >= 20:
                prices = np.array([t["price"] for t in buf])
                self._hurst_cache = self._compute_hurst(prices)
            else:
                self._hurst_cache = 0.5
            self._hurst_tick = current_tick
        return self._hurst_cache

    def hurst_passes(self, tick_buf, current_tick: int) -> tuple[bool, float]:
        H = self.hurst(tick_buf, current_tick)
        threshold = 0.44 if self.consec_losses >= 2 else 0.48
        return H < threshold, H

    # ── GARCH(1,1) volatility filter ─────────────────────────────────────

    def _fit_garch(self, tick_buf) -> float | None:
        """Fit GARCH(1,1) and return one-step-ahead σ forecast."""
        buf = list(tick_buf)[-self.GARCH_WINDOW:]
        if len(buf) < 50:
            return None
        prices   = np.array([t["price"] for t in buf])
        log_ret  = np.diff(np.log(prices + 1e-12)) * 100  # scale for arch
        if np.std(log_ret) < 1e-10:
            return None
        try:
            am    = arch_model(log_ret, vol="Garch", p=1, q=1,
                               dist="normal", rescale=False)
            res   = am.fit(disp="off", show_warning=False)
            fc    = res.forecast(horizon=1, reindex=False)
            sigma = float(np.sqrt(fc.variance.values[-1, 0])) / 100
            return sigma if np.isfinite(sigma) and sigma > 0 else None
        except Exception:
            return None

    def _ewma_sigma_fallback(self, tick_buf) -> float:
        """EWMA sigma fallback when arch is unavailable."""
        buf = list(tick_buf)[-self.GARCH_WINDOW:]
        if len(buf) < 2:
            return 1e-4
        prices  = np.array([t["price"] for t in buf])
        log_ret = np.diff(np.log(prices + 1e-12))
        alpha   = 0.06
        var     = log_ret[0] ** 2
        for r in log_ret[1:]:
            var = alpha * r**2 + (1 - alpha) * var
        return float(np.sqrt(var))

    def garch_sigma(self, tick_buf, current_tick: int) -> float | None:
        """Return cached GARCH σ, recomputing every GARCH_RECOMPUTE ticks."""
        if (self._garch_sigma_cache is None or
                current_tick - self._garch_tick >= self.GARCH_RECOMPUTE):
            if _ARCH_AVAILABLE:
                sigma = self._fit_garch(tick_buf)
            else:
                sigma = self._ewma_sigma_fallback(tick_buf)
            if sigma is not None:
                self._garch_sigma_cache = sigma
                self._garch_sigma_hist.append(sigma)
            self._garch_tick = current_tick
        return self._garch_sigma_cache

    def garch_passes(self, tick_buf, current_tick: int) -> tuple[bool, float, float]:
        """
        Pass if current GARCH σ is in the bottom 35th percentile
        of rolling σ history (calm volatility period).
        Returns (passed, sigma, percentile).
        """
        sigma = self.garch_sigma(tick_buf, current_tick)
        if sigma is None or len(self._garch_sigma_hist) < 10:
            # Not enough history yet — pass through (don't block cold start)
            return True, sigma or 0.0, 50.0
        hist        = np.array(self._garch_sigma_hist)
        percentile  = float(scipy_stats.percentileofscore(hist, sigma))
        return percentile <= 35.0, sigma, percentile

    # ── GBM barrier probability (reflection principle) ────────────────────

    @staticmethod
    def _gbm_stay_prob(sigma: float, n_ticks: int, k: float) -> float:
        """
        Analytic probability that a GBM path stays within ±k·σ·√n_ticks
        over n_ticks steps, using the reflection principle series.

        P(|W_t| < b for all t in [0,T]) = Σ_{n=-∞}^{∞} (-1)^n · Φ(...)
        Truncated to ±4 terms (converges rapidly for b/σ > 0.5).

        b     = k · σ · √n_ticks  (barrier in price-space normalised units)
        σ_t   = σ · √n_ticks      (std dev over horizon)
        """
        if sigma <= 0 or n_ticks <= 0 or k <= 0:
            return 0.5
        b     = k * sigma * math.sqrt(n_ticks)
        sigma_t = sigma * math.sqrt(n_ticks)
        if sigma_t < 1e-12:
            return 1.0
        prob = 0.0
        from scipy.special import ndtr  # standard normal CDF
        for n in range(-4, 5):
            sign = (-1) ** abs(n)
            prob += sign * (ndtr((b - 2*n*b) / sigma_t) -
                            ndtr((-b - 2*n*b) / sigma_t))
        return float(np.clip(prob, 0.0, 1.0))

    def gbm_passes(self, tick_buf, sigma_override: float | None = None
                   ) -> tuple[bool, float, float]:
        """
        Choose k=1.0 (one-sigma band) and check if stay probability ≥ 0.62.
        Returns (passed, probability, k_used).
        """
        if sigma_override is not None and sigma_override > 0:
            sigma = sigma_override
        else:
            buf   = list(tick_buf)[-50:]
            if len(buf) < 2:
                return True, 1.0, 1.0  # cold start — pass through
            prices  = np.array([t["price"] for t in buf])
            log_ret = np.diff(np.log(prices + 1e-12))
            sigma   = float(np.std(log_ret)) if len(log_ret) > 1 else 1e-4

        k    = 1.0
        prob = self._gbm_stay_prob(sigma, self.GBM_TICKS, k)
        return prob >= self.GBM_PROB_FLOOR, prob, k

    # ── Master gate ───────────────────────────────────────────────────────

    def all_pass(self, tick_buf, current_tick: int
                 ) -> tuple[bool, dict]:
        """
        Run all three filters. Returns (all_passed, details_dict).
        Fails fast — stops at first failing filter to save CPU.
        """
        details = {}

        # Filter 1: Hurst
        h_pass, H = self.hurst_passes(tick_buf, current_tick)
        details["H"] = round(H, 4)
        details["H_threshold"] = 0.44 if self.consec_losses >= 2 else 0.48
        if not h_pass:
            details["failed"] = "HURST"
            return False, details

        # Filter 2: GARCH
        g_pass, sigma, pct = self.garch_passes(tick_buf, current_tick)
        details["garch_sigma"]  = round(sigma, 8)
        details["garch_pct"]    = round(pct, 1)
        if not g_pass:
            details["failed"] = "GARCH"
            return False, details

        # Filter 3: GBM
        gbm_pass, prob, k = self.gbm_passes(tick_buf, sigma_override=sigma)
        details["gbm_prob"] = round(prob, 4)
        details["gbm_k"]    = round(k, 2)
        if not gbm_pass:
            details["failed"] = "GBM"
            return False, details

        details["failed"] = None
        return True, details

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    cfg    = CONFIG
    symbol = cfg["symbol"]

    log.info("=" * 65)
    log.info("  DERIV RANGE BOT  v9  —  Pure ContainmentEstimator")
    log.info("  Symbol  : %s", symbol)
    log.info("  EV floor: %.3f  min EV: %.3f  (48%% payout break-even)",
             cfg["ev_confidence_floor"], cfg["min_ev_threshold"])
    log.info("  Staking : %.1f%% of balance  (min $%.2f)  — calibrated for $1 account",
             cfg["stake_pct"]*100, cfg["kelly_min_stake"])
    log.info("  Persist : %d consecutive ticks above floor",
             cfg["signal_persistence_ticks"])
    log.info("=" * 65)

    if not cfg.get("api_token"):
        log.error("DERIV_API_TOKEN not set. Exiting.")
        sys.exit(1)

    # ── Phase 1: Load or fetch historical ticks ────────────────────────────
    log.info("\n>> PHASE 1 — Historical tick collection")
    data_path = os.path.join(cfg["data_dir"], f"ticks_{symbol}.csv")
    existing  = pd.read_csv(data_path) if os.path.isfile(data_path) else pd.DataFrame()

    done = threading.Event()
    HistoricalCollector(symbol, cfg, done, existing).start()
    done.wait()

    df = pd.read_csv(data_path)
    if len(df) < cfg["min_ticks"]:
        log.warning("[Main] Only %d ticks (need %d). "
                    "Will trade once buffer fills live.", len(df), cfg["min_ticks"])

    # Seed tick buffer from historical data (most recent max_buf ticks)
    initial_ticks = df.tail(5000).to_dict("records")
    log.info("[Main] Seeding tick buffer with %d historical ticks.",
             len(initial_ticks))

    # ── Phase 2: Start live trading ────────────────────────────────────────
    log.info("\n>> PHASE 2 — Starting live trading on %s", symbol)

    staker  = KellyStaker(cfg)
    thr     = AdaptiveThreshold(cfg)
    sprt    = SPRTMonitor(
        p0=cfg["sprt_p0"], p1=cfg["sprt_p1"],
        alpha=cfg["sprt_alpha"], beta=cfg["sprt_beta"],
    )
    tlog    = TradeLogger(cfg["trade_log"])
    trader  = LiveTrader(cfg, initial_ticks, staker, thr, sprt, tlog)

    # Graceful Ctrl+C
    import signal as _sig
    def _shutdown(s, f):
        log.info("\n[Main] Shutting down ...")
        trader.stop()
    _sig.signal(_sig.SIGINT,  _shutdown)
    _sig.signal(_sig.SIGTERM, _shutdown)

    trader.run()

    log.info("\n[Main] Session complete.")
    log.info("  Trades    : %d", trader.total)
    log.info("  Win rate  : %.1f%%", trader.wins/trader.total*100
             if trader.total else 0)
    log.info("  P&L       : %+.2f", trader.session_pnl)
    log.info("  SPRT      : %s", sprt.summary())
    log.info("  Trade log : %s", cfg["trade_log"])


if __name__ == "__main__":
    main()
