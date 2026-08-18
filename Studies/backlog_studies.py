"""
backlog_studies.py — standalone backtest of three remaining "have the
data already" backlog strategies:

  Study A  Concentration ratio extremes  — top-4-trader NET concentration
                                            (Conc Net 4 Long/Short, % of
                                            OI) at a point-in-time extreme
                                            -> squeeze/manipulation risk,
                                            fade it. KC/CC/SB/CT only
                                            (concentration data isn't
                                            populated for RC/LCC/LSU in
                                            this parquet).
  Study B  Old-crop vs new-crop divergence — MM Net positioning skew
                                              between Old-crop and
                                              Other(new)-crop contracts.
                                              Tested against the OUTRIGHT
                                              continuous price (proxy —
                                              the real test would use the
                                              actual old/new calendar
                                              spread, which isn't
                                              available as a clean series
                                              here). All 7 commodities.
  Study C  Index/Swap mechanical flow    — CIT's Index Long/Short (pure
                                            index-fund category) + Disagg's
                                            Swap Long/Short (closest Disagg
                                            equivalent) as one "mechanical,
                                            non-discretionary flow"
                                            category, pooled across all 7.
                                            Tests both positioning
                                            extremity AND price-impact
                                            (does mechanical flow move
                                            price more cleanly than
                                            discretionary spec flow did).

Run:  python backlog_studies.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DB_DIR      = Path(__file__).resolve().parent.parent / "Database"
CIT_FILE    = DB_DIR / "cot_cit.parquet"
DISAGG_FILE = DB_DIR / "cot_disagg_fut.parquet"
ROLLEX_DIR  = DB_DIR / "Rollex"
ROLLEX_MAP  = {"KC": "rollex_KC.parquet", "CC": "rollex_CC.parquet",
               "SB": "rollex_SB.parquet", "CT": "rollex_CT.parquet",
               "RC": "rollex_RC.parquet", "LCC": "rollex_LCC.parquet",
               "LSU": "rollex_LSU.parquet"}
CIT_SYMS    = ["KC", "CC", "SB", "CT"]
ALL_SYMS    = ["KC", "CC", "SB", "CT", "RC", "LCC", "LSU"]

MIN_HISTORY_WEEKS = 104
N_BOOT   = 5000
BLOCK    = 8
COST_BPS = 4.0
HOLD     = 4
RNG      = np.random.default_rng(29)


# ══════════════════════════════════════════════════════════════════════════
# Shared infra (same as the other Studies scripts)
# ══════════════════════════════════════════════════════════════════════════
def load_rollex(sym: str) -> pd.DataFrame:
    rx = pd.read_parquet(ROLLEX_DIR / ROLLEX_MAP[sym], columns=["rollex_px"])
    rx.index = pd.to_datetime(rx.index)
    rx.index.name = "Date"
    return rx.reset_index().sort_values("Date")


def expanding_percentile(s: pd.Series, min_periods=MIN_HISTORY_WEEKS) -> np.ndarray:
    vals = s.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(min_periods - 1, len(vals)):
        window = vals[: i + 1]
        out[i] = (window <= vals[i]).mean() * 100.0
    return out


def forward_return(px: pd.Series, n: int) -> np.ndarray:
    return (px.shift(-n).values / px.values - 1.0) * 100.0


def block_bootstrap_pvalue(mask: np.ndarray, y: np.ndarray, block=BLOCK, n_boot=N_BOOT) -> tuple:
    valid = ~np.isnan(y) & ~np.isnan(mask.astype(float))
    y, mask = y[valid], mask[valid]
    if mask.sum() < 8 or (~mask).sum() < 8:
        return np.nan, np.nan
    obs_diff = y[mask].mean() - y[~mask].mean()
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        ys, ms = y[idx], mask[idx]
        if ms.sum() == 0 or (~ms).sum() == 0:
            diffs[b] = np.nan
            continue
        diffs[b] = ys[ms].mean() - ys[~ms].mean()
    diffs = diffs[~np.isnan(diffs)]
    p = float((np.abs(diffs) >= abs(obs_diff)).mean())
    return obs_diff, p


def pooled_block_bootstrap_pvalue(groups: list, block=BLOCK, n_boot=N_BOOT) -> tuple:
    clean = []
    for mask, y in groups:
        valid = ~np.isnan(y) & ~np.isnan(mask.astype(float))
        m, yv = mask[valid], y[valid]
        if len(yv) > 0:
            clean.append((m, yv))
    if not clean:
        return np.nan, np.nan, 0
    all_mask = np.concatenate([m for m, _ in clean])
    all_y = np.concatenate([y for _, y in clean])
    if all_mask.sum() < 8 or (~all_mask).sum() < 8:
        return np.nan, np.nan, int(all_mask.sum())
    obs_diff = all_y[all_mask].mean() - all_y[~all_mask].mean()
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        bm_parts, by_parts = [], []
        for m, y in clean:
            n = len(y)
            if n <= block:
                bm_parts.append(m); by_parts.append(y); continue
            n_blocks = int(np.ceil(n / block))
            starts = RNG.integers(0, n - block, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            bm_parts.append(m[idx]); by_parts.append(y[idx])
        bm, by = np.concatenate(bm_parts), np.concatenate(by_parts)
        if bm.sum() == 0 or (~bm).sum() == 0:
            diffs[b] = np.nan
            continue
        diffs[b] = by[bm].mean() - by[~bm].mean()
    diffs = diffs[~np.isnan(diffs)]
    p = float((np.abs(diffs) >= abs(obs_diff)).mean())
    return obs_diff, p, int(all_mask.sum())


def bh_correct(pvals: list) -> list:
    """Benjamini-Hochberg FDR correction. NaN p-values (from zero-event
    buckets) are excluded from the correction entirely, not just sorted to
    the end — np.minimum.accumulate propagates a single NaN through the
    whole cumulative-min chain, silently turning every corrected p-value
    into NaN if left in."""
    pvals = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    out = np.full(len(pvals), np.nan)
    if valid.sum() == 0:
        return list(out)
    vp = pvals[valid]
    n = len(vp)
    order = np.argsort(vp)
    ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    adj = vp * n / ranks
    corrected_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    corrected = np.empty(n)
    corrected[order] = np.clip(corrected_sorted, 0, 1)
    out[valid] = corrected
    return list(out)


def backtest_stats(bt: pd.DataFrame, split_frac=0.7) -> dict:
    n = len(bt)
    cut = int(n * split_frac)
    out = {}
    for label, sub in [("full", bt), ("in-sample (first 70%)", bt.iloc[:cut]),
                        ("out-of-sample (last 30%)", bt.iloc[cut:])]:
        r = sub["ret"].values
        std = r.std(ddof=1)
        sharpe = (r.mean() / std * np.sqrt(52)) if std > 0 else np.nan
        equity = (1 + r).cumprod()
        dd = (equity / np.maximum.accumulate(equity) - 1).min() if len(equity) else np.nan
        out[label] = dict(weeks=len(sub), total_ret_pct=(equity[-1] - 1) * 100 if len(equity) else np.nan,
                           ann_sharpe=sharpe, max_dd_pct=dd * 100 if pd.notna(dd) else np.nan)
    return out


def lagged_backtest(px: np.ndarray, dates: np.ndarray, signal: np.ndarray, hold=HOLD) -> pd.DataFrame:
    """position = signal held for `hold` weeks after it fires, lagged 1wk entry."""
    n = len(px)
    pos = np.zeros(n)
    for i in np.where(signal != 0)[0]:
        entry = i + 1
        if entry < n:
            pos[entry: min(entry + hold, n)] = signal[i]
    ret_1w = np.nan_to_num(np.r_[np.nan, px[1:] / px[:-1] - 1.0])
    strat_ret = pos * ret_1w
    cost = np.r_[0, np.abs(np.diff(pos))] * (COST_BPS / 10000.0)
    return pd.DataFrame(dict(Date=dates, ret=strat_ret - cost, pos=pos))


# ══════════════════════════════════════════════════════════════════════════
# Study A — Concentration ratio extremes (KC/CC/SB/CT)
# ══════════════════════════════════════════════════════════════════════════
def load_concentration(sym: str) -> pd.DataFrame:
    dis = pd.read_parquet(DISAGG_FILE)
    dis = dis[(dis["Commodity"] == sym) & (dis["Crop"] == "All")].sort_values("Date").reset_index(drop=True)
    dis["Date"] = pd.to_datetime(dis["Date"])
    rx = load_rollex(sym)
    d = pd.merge_asof(dis[["Date", "Conc Net 4 Long", "Conc Net 4 Short"]], rx, on="Date", direction="backward")
    return d.dropna().reset_index(drop=True)


def study_a_concentration(horizons=(1, 2, 4, 8)) -> tuple:
    rows, data_by_sym = [], {}
    for sym in CIT_SYMS:
        d = load_concentration(sym)
        data_by_sym[sym] = d
        pctL = expanding_percentile(d["Conc Net 4 Long"])
        pctS = expanding_percentile(d["Conc Net 4 Short"])
        for h in horizons:
            fwd = forward_return(d["rollex_px"], h)
            hi_long = pctL >= 90
            diff, p = block_bootstrap_pvalue(hi_long, fwd)
            rows.append(dict(symbol=sym, side="conc_long_extreme", horizon=h, n=int(np.nansum(hi_long)),
                              mean_diff_pct=diff, p=p))
            hi_short = pctS >= 90
            diff, p = block_bootstrap_pvalue(hi_short, fwd)
            rows.append(dict(symbol=sym, side="conc_short_extreme", horizon=h, n=int(np.nansum(hi_short)),
                              mean_diff_pct=diff, p=p))

    # pooled
    for h in horizons:
        groups_long, groups_short = [], []
        for sym, d in data_by_sym.items():
            pctL = expanding_percentile(d["Conc Net 4 Long"])
            pctS = expanding_percentile(d["Conc Net 4 Short"])
            fwd = forward_return(d["rollex_px"], h)
            groups_long.append((pctL >= 90, fwd))
            groups_short.append((pctS >= 90, fwd))
        diff, p, n = pooled_block_bootstrap_pvalue(groups_long)
        rows.append(dict(symbol="POOLED(4)", side="conc_long_extreme", horizon=h, n=n, mean_diff_pct=diff, p=p))
        diff, p, n = pooled_block_bootstrap_pvalue(groups_short)
        rows.append(dict(symbol="POOLED(4)", side="conc_short_extreme", horizon=h, n=n, mean_diff_pct=diff, p=p))

    # backtest: fade extreme concentration (short if long-conc extreme, long if short-conc extreme)
    per_sym_bt = {}
    for sym, d in data_by_sym.items():
        pctL = expanding_percentile(d["Conc Net 4 Long"])
        pctS = expanding_percentile(d["Conc Net 4 Short"])
        sig = np.where(pctL >= 90, -1.0, np.where(pctS >= 90, 1.0, 0.0))
        per_sym_bt[sym] = lagged_backtest(d["rollex_px"].values, d["Date"].values, sig)
    merged = None
    for sym, bt in per_sym_bt.items():
        bt = bt.rename(columns={"ret": f"ret_{sym}", "pos": f"pos_{sym}"})
        merged = bt if merged is None else merged.merge(bt, on="Date", how="inner")
    ret_cols = [c for c in merged.columns if c.startswith("ret_")]
    pooled_bt = pd.DataFrame(dict(Date=merged["Date"], ret=merged[ret_cols].mean(axis=1)))
    return rows, per_sym_bt, pooled_bt


# ══════════════════════════════════════════════════════════════════════════
# Study B — Old-crop vs new-crop divergence (all 7)
# ══════════════════════════════════════════════════════════════════════════
def load_oldnew(sym: str) -> pd.DataFrame:
    dis = pd.read_parquet(DISAGG_FILE)
    dis = dis[dis["Commodity"] == sym].sort_values("Date").reset_index(drop=True)
    dis["Date"] = pd.to_datetime(dis["Date"])
    old = dis[dis["Crop"] == "Old"][["Date", "MM Long", "MM Short"]].rename(
        columns={"MM Long": "OldL", "MM Short": "OldS"})
    new = dis[dis["Crop"] == "Other"][["Date", "MM Long", "MM Short"]].rename(
        columns={"MM Long": "NewL", "MM Short": "NewS"})
    rx = load_rollex(sym)
    d = old.merge(new, on="Date", how="inner").merge(rx, on="Date", how="inner")
    d["OldNet"] = d["OldL"] - d["OldS"]
    d["NewNet"] = d["NewL"] - d["NewS"]
    return d.dropna().reset_index(drop=True)


def study_b_oldnew(horizons=(1, 2, 4, 8)) -> tuple:
    rows, data_by_sym = [], {}
    for sym in ALL_SYMS:
        d = load_oldnew(sym)
        if len(d) < MIN_HISTORY_WEEKS + 20:
            continue
        data_by_sym[sym] = d
        pctOld = expanding_percentile(d["OldNet"])
        pctNew = expanding_percentile(d["NewNet"])
        divergence = pctOld - pctNew
        for h in horizons:
            fwd = forward_return(d["rollex_px"], h)
            bull = divergence <= -60   # new-crop relatively more long than old-crop
            bear = divergence >= 60
            diff, p = block_bootstrap_pvalue(bull, fwd)
            rows.append(dict(symbol=sym, side="bullish(new>old)", horizon=h, n=int(np.nansum(bull)),
                              mean_diff_pct=diff, p=p))
            diff, p = block_bootstrap_pvalue(bear, fwd)
            rows.append(dict(symbol=sym, side="bearish(old>new)", horizon=h, n=int(np.nansum(bear)),
                              mean_diff_pct=diff, p=p))

    for h in horizons:
        groups_bull, groups_bear = [], []
        for sym, d in data_by_sym.items():
            pctOld = expanding_percentile(d["OldNet"])
            pctNew = expanding_percentile(d["NewNet"])
            divergence = pctOld - pctNew
            fwd = forward_return(d["rollex_px"], h)
            groups_bull.append((divergence <= -60, fwd))
            groups_bear.append((divergence >= 60, fwd))
        diff, p, n = pooled_block_bootstrap_pvalue(groups_bull)
        rows.append(dict(symbol="POOLED(7)", side="bullish(new>old)", horizon=h, n=n, mean_diff_pct=diff, p=p))
        diff, p, n = pooled_block_bootstrap_pvalue(groups_bear)
        rows.append(dict(symbol="POOLED(7)", side="bearish(old>new)", horizon=h, n=n, mean_diff_pct=diff, p=p))

    per_sym_bt = {}
    for sym, d in data_by_sym.items():
        pctOld = expanding_percentile(d["OldNet"])
        pctNew = expanding_percentile(d["NewNet"])
        divergence = pctOld - pctNew
        sig = np.where(divergence <= -60, 1.0, np.where(divergence >= 60, -1.0, 0.0))
        per_sym_bt[sym] = lagged_backtest(d["rollex_px"].values, d["Date"].values, sig)
    merged = None
    for sym, bt in per_sym_bt.items():
        bt = bt.rename(columns={"ret": f"ret_{sym}", "pos": f"pos_{sym}"})
        merged = bt if merged is None else merged.merge(bt, on="Date", how="inner")
    ret_cols = [c for c in merged.columns if c.startswith("ret_")]
    pooled_bt = pd.DataFrame(dict(Date=merged["Date"], ret=merged[ret_cols].mean(axis=1)))
    return rows, per_sym_bt, pooled_bt


# ══════════════════════════════════════════════════════════════════════════
# Study C — Index/Swap mechanical flow (all 7, pooled category)
# ══════════════════════════════════════════════════════════════════════════
def load_mech_flow(sym: str) -> pd.DataFrame:
    rx = load_rollex(sym)
    if sym in CIT_SYMS:
        cot = pd.read_parquet(CIT_FILE)
        cot = cot[cot["Commodity"] == sym].sort_values("Date").reset_index(drop=True)
        cot["Date"] = pd.to_datetime(cot["Date"])
        cot = cot.rename(columns={"Index Long": "MechLong", "Index Short": "MechShort"})
    else:
        cot = pd.read_parquet(DISAGG_FILE)
        cot = cot[(cot["Commodity"] == sym) & (cot["Crop"] == "All")].sort_values("Date").reset_index(drop=True)
        cot["Date"] = pd.to_datetime(cot["Date"])
        cot = cot.rename(columns={"Swap Long": "MechLong", "Swap Short": "MechShort"})
    d = pd.merge_asof(cot[["Date", "MechLong", "MechShort"]], rx, on="Date", direction="backward")
    return d.dropna().reset_index(drop=True)


def study_c_mechflow(horizons=(1, 2, 4, 8)) -> tuple:
    rows, data_by_sym = [], {}
    for sym in ALL_SYMS:
        d = load_mech_flow(sym)
        data_by_sym[sym] = d
        pctL = expanding_percentile(d["MechLong"])
        pctS = expanding_percentile(d["MechShort"])
        for h in horizons:
            fwd = forward_return(d["rollex_px"], h)
            hi_long = pctL >= 90
            diff, p = block_bootstrap_pvalue(hi_long, fwd)
            rows.append(dict(symbol=sym, side="mech_long_extreme", horizon=h, n=int(np.nansum(hi_long)),
                              mean_diff_pct=diff, p=p))
            hi_short = pctS >= 90
            diff, p = block_bootstrap_pvalue(hi_short, fwd)
            rows.append(dict(symbol=sym, side="mech_short_extreme", horizon=h, n=int(np.nansum(hi_short)),
                              mean_diff_pct=diff, p=p))

    for h in horizons:
        groups_long, groups_short = [], []
        for sym, d in data_by_sym.items():
            pctL = expanding_percentile(d["MechLong"])
            pctS = expanding_percentile(d["MechShort"])
            fwd = forward_return(d["rollex_px"], h)
            groups_long.append((pctL >= 90, fwd))
            groups_short.append((pctS >= 90, fwd))
        diff, p, n = pooled_block_bootstrap_pvalue(groups_long)
        rows.append(dict(symbol="POOLED(7)", side="mech_long_extreme", horizon=h, n=n, mean_diff_pct=diff, p=p))
        diff, p, n = pooled_block_bootstrap_pvalue(groups_short)
        rows.append(dict(symbol="POOLED(7)", side="mech_short_extreme", horizon=h, n=n, mean_diff_pct=diff, p=p))

    # price-impact regression: does fresh mechanical flow move price? (same-week OLS, block-bootstrap on beta)
    impact_rows = []
    for sym, d in data_by_sym.items():
        dMech = (d["MechLong"] - d["MechShort"]).diff().values / 1000.0
        dPx = d["rollex_px"].pct_change().values * 100
        mask = ~(np.isnan(dMech) | np.isnan(dPx))
        x, y = dMech[mask], dPx[mask]
        if len(x) < 30:
            continue
        X = np.column_stack([x, np.ones_like(x)])
        coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        beta = coef[0]
        # block bootstrap on beta
        n = len(x)
        n_blocks = int(np.ceil(n / BLOCK))
        betas = np.empty(2000)
        for b in range(2000):
            starts = RNG.integers(0, n - BLOCK, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
            Xi, yi = X[idx], y[idx]
            c2, _, r2, _ = np.linalg.lstsq(Xi, yi, rcond=None)
            betas[b] = c2[0]
        se = betas.std()
        p_beta = float((np.abs(betas - betas.mean()) >= abs(beta - betas.mean())).mean())
        impact_rows.append(dict(symbol=sym, beta_px_per_klot=beta, se=se, p=p_beta, n=n))

    per_sym_bt = {}
    for sym, d in data_by_sym.items():
        pctL = expanding_percentile(d["MechLong"])
        pctS = expanding_percentile(d["MechShort"])
        sig = np.where(pctL >= 90, -1.0, np.where(pctS >= 90, 1.0, 0.0))  # fade extremity
        per_sym_bt[sym] = lagged_backtest(d["rollex_px"].values, d["Date"].values, sig)
    merged = None
    for sym, bt in per_sym_bt.items():
        bt = bt.rename(columns={"ret": f"ret_{sym}", "pos": f"pos_{sym}"})
        merged = bt if merged is None else merged.merge(bt, on="Date", how="inner")
    ret_cols = [c for c in merged.columns if c.startswith("ret_")]
    pooled_bt = pd.DataFrame(dict(Date=merged["Date"], ret=merged[ret_cols].mean(axis=1)))
    return rows, impact_rows, per_sym_bt, pooled_bt


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════
def print_backtest(label, per_sym_bt, pooled_bt):
    print(f"\n  {label} — per-symbol + pooled backtest ({HOLD}wk hold, {COST_BPS}bps cost, lagged 1wk):")
    for sym, bt in per_sym_bt.items():
        s = backtest_stats(bt)["full"]
        print(f"    {sym:5s}  weeks={s['weeks']:4d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
              f"ann_sharpe={s['ann_sharpe']:+.2f}  max_dd={s['max_dd_pct']:6.1f}%")
    stats = backtest_stats(pooled_bt)
    for lbl, s in stats.items():
        print(f"    POOLED {lbl:22s}  weeks={s['weeks']:4d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
              f"ann_sharpe={s['ann_sharpe']:+.2f}  max_dd={s['max_dd_pct']:6.1f}%")


def main():
    all_rows = []

    print(f"{'='*70}\nSTUDY A — Concentration ratio extremes (KC/CC/SB/CT)\n{'='*70}")
    rows_a, bt_a, pooled_a = study_a_concentration()
    for r in rows_a:
        r["study"] = "A"
    all_rows += rows_a
    print_backtest("Study A", bt_a, pooled_a)

    print(f"\n\n{'='*70}\nSTUDY B — Old-crop vs new-crop divergence (all 7)\n{'='*70}")
    rows_b, bt_b, pooled_b = study_b_oldnew()
    for r in rows_b:
        r["study"] = "B"
    all_rows += rows_b
    print_backtest("Study B", bt_b, pooled_b)

    print(f"\n\n{'='*70}\nSTUDY C — Index/Swap mechanical flow (all 7, pooled category)\n{'='*70}")
    rows_c, impact_c, bt_c, pooled_c = study_c_mechflow()
    for r in rows_c:
        r["study"] = "C"
    all_rows += rows_c
    print("\n  Price-impact regression (dMechNet ~ dPx, same-week, block-bootstrap on beta):")
    for r in impact_c:
        sig = "significant" if r["p"] < 0.05 else "not significant"
        print(f"    {r['symbol']:5s}  beta={r['beta_px_per_klot']:+.3f}%/k-lot  se={r['se']:.3f}  "
              f"p={r['p']:.3f} ({sig})  n={r['n']}")
    print_backtest("Study C", bt_c, pooled_c)

    df = pd.DataFrame(all_rows)
    df["p_bh"] = bh_correct(df["p"].tolist())
    print(f"\n\n{'='*70}\nALL BUCKET TESTS  (p_bh = Benjamini-Hochberg corrected across "
          f"Studies A+B+C combined)\n{'='*70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df[["study", "symbol", "side", "horizon", "n", "mean_diff_pct", "p", "p_bh"]]
              .round(3).to_string(index=False))

    sig = df[df["p_bh"] <= 0.10]
    print(f"\n{'='*70}\nSURVIVES q<=0.10 after FDR correction: {len(sig)} / {len(df)}\n{'='*70}")
    if len(sig):
        print(sig[["study", "symbol", "side", "horizon", "mean_diff_pct", "p_bh"]].to_string(index=False))

    out_path = Path(__file__).parent / "backlog_studies_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
