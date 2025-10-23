"""
Bachelier-based market making strategy for the Cornell Derivatives Case.

Expected environment
--------------------
- The autograder injects an AbstractTradingStrategy to subclass.
- marketplace.get_products() -> List[Product] (with .product_id)
- marketplace.get_die_num_faces() -> int (optional helper)
- training_rolls: 10,000 historical rolls (iterable)
- current_rolls: rolls seen so far this round (iterable)
- my_trades: either a list[Trade] or an object with get_position(...)
- round_info: dict with keys like 'current_sub_round', 'num_sub_rounds', 'round_id'
"""

from typing import Any, Dict, Tuple, List
from math import erf, sqrt, exp, pi
# Import path based on the provided example spec. This should exist on Gradescope.
from autograder.sdk.strategy_interface import AbstractTradingStrategy


# ---------- math utils: normal PDF/CDF ----------
def _phi(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ---------- Bachelier option pricers (additive model) ----------
def _bachelier_call(mu: float, sigma: float, K: float) -> float:
    if sigma <= 1e-12:
        return max(0.0, mu - K)
    d = (mu - K) / sigma
    return (mu - K) * _Phi(d) + sigma * _phi(d)


def _bachelier_put(mu: float, sigma: float, K: float) -> float:
    if sigma <= 1e-12:
        return max(0.0, K - mu)
    d = (mu - K) / sigma
    return (K - mu) * _Phi(-d) + sigma * _phi(d)


# ---------- helper: pooled per-die stats ----------
def _pooled_mean_var_per_die(training_rolls: Any, observed_rolls: Any) -> Tuple[float, float]:
    train = list(training_rolls) if hasattr(training_rolls, "__iter__") else []
    obs = list(observed_rolls) if hasattr(observed_rolls, "__iter__") else []

    n_t = len(train)
    n_o = len(obs)
    n = n_t + n_o

    if n == 0:
        return (0.0, 0.0)

    sum_t = float(sum(train)) if n_t else 0.0
    sum_o = float(sum(obs)) if n_o else 0.0
    mean_t = (sum_t / n_t) if n_t else 0.0
    mean_o = (sum_o / n_o) if n_o else 0.0

    if n_t > 1:
        var_t = sum((x - mean_t) * (x - mean_t) for x in train) / (n_t - 1)
    else:
        var_t = 0.0
    if n_o > 1:
        var_o = sum((x - mean_o) * (x - mean_o) for x in obs) / (n_o - 1)
    else:
        var_o = 0.0

    mean = (sum_t + sum_o) / n

    ssq = 0.0
    if n_t > 1:
        ssq += (n_t - 1) * var_t + (n_t * (mean_t - mean) * (mean_t - mean))
    elif n_t == 1:
        ssq += (n_t * (mean_t - mean) * (mean_t - mean))
    if n_o > 1:
        ssq += (n_o - 1) * var_o + (n_o * (mean_o - mean) * (mean_o - mean))
    elif n_o == 1:
        ssq += (n_o * (mean_o - mean) * (mean_o - mean))

    var = (ssq / (n - 1)) if n > 1 else 0.0
    return mean, var


# ---------- helper: parse product_id ----------
def _parse_product_id(product_id: str):
    parts = product_id.split(",")
    if len(parts) < 3 or parts[0] not in ("S", "P"):
        return None, None, None
    kind = parts[1]
    strike = None
    if kind in ("C", "P") and len(parts) >= 4:
        try:
            strike = float(parts[2])
        except Exception:
            strike = None
    try:
        expiry = int(parts[3])
    except Exception:
        expiry = None
    return kind, expiry, strike


# ---------- helper: inventory from my_trades ----------
def _compute_inventory_map(my_trades: Any, team_name: str) -> Dict[str, float]:
    inv: Dict[str, float] = {}
    if hasattr(my_trades, "__iter__"):
        try:
            for t in my_trades:
                pid = getattr(t, "product_id", None)
                if not pid:
                    continue
                q = float(getattr(t, "quantity", 0.0) or 0.0)
                buyer = getattr(t, "buyer_id", "")
                seller = getattr(t, "seller_id", "")
                if buyer == team_name:
                    inv[pid] = inv.get(pid, 0.0) + q
                if seller == team_name:
                    inv[pid] = inv.get(pid, 0.0) - q
        except TypeError:
            pass
    return inv


# ---------- BachelierStrategy with print statements ----------
class BachelierStrategy(AbstractTradingStrategy):
    def __init__(self):
        # ---------- Global / common tuning parameters ----------
        self.min_tick: float = 0.01  # floor for quotes — Increase→widens (fewer trades), Decrease→tightens (more trades)
        self.base_spread: float = 0.40  # additive spread base — Increase→widens, Decrease→tightens (TIGHTENED from 1.0)
        self.sigma_spread_mult: float = 0.008  # sensitivity of spread to σ — Increase→widens more when σ high, Decrease→tightens (TIGHTENED from 0.02)
        self.time_decay_per_sub: float = 0.15  # how fast spreads tighten across subrounds — Increase→tighter later, Decrease→wider longer (INCREASED from 0.10)
        self.min_decay: float = 0.15  # lower bound on decay multiplier — Increase→wider late-round floor, Decrease→tighter late-round spreads (REDUCED from 0.25)
        self.inv_skew_per_unit: float = 0.03  # inventory shading slope — Increase→more shading (widens on heavy side), Decrease→less shading (tighter/symmetric) (REDUCED from 0.05)

        # ---------- Options-specific tuning ----------
        self.option_spread_mult: float = 0.85  # multiplicative spread on options — Increase→widens options, Decrease→tightens (TIGHTENED from 1.25)
        self.option_min_spread: float = 0.01  # absolute floor for option spread — Increase→no tighter than floor, Decrease→allows tighter (TIGHTENED from 0.02)
        self.min_model_edge: float = 0.03  # minimum model edge before showing a side — Increase→fewer fills (safer), Decrease→more fills (riskier) (REDUCED from 0.10)

        # ---------- Aggressive options mode parameters ----------
        self.aggressive_options: bool = True  # False→normal mode, True→explicitly aggressive (tighter spreads, inward push) (ENABLED)
        self.option_trade_boost: float = 0.50  # multiplicative shrink of spread when "cheap" — Increase→tighter when cheap, Decrease→less tightening (INCREASED from 0.35)
        self.option_aggressive_push: float = 0.35  # inward pull fraction of spread — Increase→more inward push, Decrease→less push (INCREASED from 0.25)
        self.option_cheap_price_threshold: float = 5.0  # FV threshold for cheap options — Increase→more cheap options (more aggression) (INCREASED from 3.0)
        self.option_sigma_cheap_threshold: float = 80.0  # σ threshold for cheap options — Increase→more cheap options, Decrease→fewer (INCREASED from 50.0)
        self.option_push_keep_spread_frac: float = 0.05  # minimum residual spread fraction after push — Increase→wider residual (safer), Decrease→tighter final spread (REDUCED from 0.10)

        # ---------- Environment defaults ----------
        self.team_name: str = "Team"
        self.num_sub_rounds: int = 10

    def on_game_start(self, config: Dict[str, Any]) -> None:
        self.team_name = config.get("team_name", "Team")
        self.num_sub_rounds = int(config.get("num_sub_rounds", 10) or 10)
        for k in (
            "min_tick","base_spread","sigma_spread_mult","time_decay_per_sub","min_decay",
            "inv_skew_per_unit","option_spread_mult","option_min_spread","min_model_edge",
            "aggressive_options","option_trade_boost","option_aggressive_push",
            "option_cheap_price_threshold","option_sigma_cheap_threshold",
            "option_push_keep_spread_frac"
        ):
            if k in config:
                setattr(self, k, type(getattr(self, k))(config[k]))

    def make_market(
        self,
        *,
        marketplace: Any,
        training_rolls: Any,
        my_trades: Any,
        current_rolls: Any,
        round_info: Any
    ) -> Dict[str, Tuple[float, float]]:

        mean_per_die, var_per_die = _pooled_mean_var_per_die(training_rolls, current_rolls)
        observed_sum = float(sum(current_rolls)) if hasattr(current_rolls, "__iter__") else 0.0
        current_sub = int(round_info.get("current_sub_round", 1) or 1)
        total_dice_in_round = 20000
        dice_seen = min(total_dice_in_round, current_sub * 2000)
        dice_remaining = max(0, total_dice_in_round - dice_seen)
        mu_total = observed_sum + dice_remaining * mean_per_die
        sigma_total = sqrt(max(0.0, dice_remaining * max(0.0, var_per_die)))
        inv_map = _compute_inventory_map(my_trades, self.team_name)
        decay_mult = max(self.min_decay, 1.1 - self.time_decay_per_sub * current_sub)
        quotes: Dict[str, Tuple[float, float]] = {}
        products = marketplace.get_products()

        for product in products:
            pid = getattr(product, "product_id", None) or getattr(product, "id", None)
            if not pid:
                continue

            kind, expiry, strike = _parse_product_id(pid)
            if kind is None:
                continue

            if kind == "F":
                fair = mu_total
            elif kind == "C":
                if strike is None:
                    continue
                fair = _bachelier_call(mu_total, sigma_total, strike)
            elif kind == "P":
                if strike is None:
                    continue
                fair = _bachelier_put(mu_total, sigma_total, strike)
            else:
                continue

            spread = self.base_spread + (self.sigma_spread_mult * sigma_total)
            spread *= decay_mult
            if kind in ("C", "P"):
                spread *= max(0.0, self.option_spread_mult)
                spread = max(spread, self.option_min_spread)
            spread = max(spread, self.min_tick)

            inv = inv_map.get(pid, 0.0)
            skew = self.inv_skew_per_unit * inv
            adjusted_mid = fair - skew

            push_amt = 0.0
            if kind in ("C", "P") and self.aggressive_options:
                is_cheap_by_price = fair <= self.option_cheap_price_threshold
                is_cheap_by_sigma = sigma_total <= self.option_sigma_cheap_threshold
                if is_cheap_by_price or is_cheap_by_sigma:
                    boost = max(0.0, min(1.0, self.option_trade_boost))
                    spread = max(self.min_tick, spread * (1.0 - boost))
                    push_frac = max(0.0, min(1.0, self.option_aggressive_push))
                    push_amt = 0.5 * spread * push_frac

            bid = max(self.min_tick, adjusted_mid - 0.5 * spread)
            ask = adjusted_mid + 0.5 * spread

            if push_amt > 0.0:
                bid += push_amt
                ask -= push_amt
                keep_frac = max(0.0, min(1.0, self.option_push_keep_spread_frac))
                min_keep = max(self.min_tick, keep_frac * spread)
                if ask - bid < min_keep:
                    mid = 0.5 * (bid + ask)
                    bid = max(self.min_tick, mid - 0.5 * min_keep)
                    ask = mid + 0.5 * min_keep
            else:
                edge = max(self.min_tick, float(self.min_model_edge))
                bid = min(bid, fair - edge)
                ask = max(ask, fair + edge)
                if ask - bid < max(self.min_tick, 0.25 * self.min_tick):
                    bid = fair - edge
                    ask = fair + edge

            if bid < ask and ask > self.min_tick:
                bid = round(max(self.min_tick, bid), 4)
                ask = round(ask, 4)
                quotes[pid] = (bid, ask)

                # ---------- PRINT STATEMENTS ----------
                # if kind == "F":
                #     print(f"[FUTURE] {pid}: fair={fair:.4f}, bid={bid:.4f}, ask={ask:.4f}")
                # elif kind in ("C", "P"):
                #     mode = "AGGR" if self.aggressive_options else "NORMAL"
                #     print(
                #         f"[OPTION-{kind}] {pid}: strike={strike}, fair={fair:.4f}, "
                #         f"bid={bid:.4f}, ask={ask:.4f}, mode={mode}"
                #     )

        return quotes

    def on_round_end(self, result: Dict[str, Any]) -> None:
        return

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        return
