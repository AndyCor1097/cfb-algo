"""
backtest.py — Phase 5: True walk-forward backtest
- Train on season N-1, test on season N
- Within each season, simulate week by week
- No future data ever touches training
"""

import pandas as pd, numpy as np, os
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.dirname(__file__)

HOME_FEATS = ["sp_off_h","sp_def_a","elo_h","talent_h","ret_off_h","ret_def_a","portal_net_h","neutral_site","home_field","week"]
AWAY_FEATS = ["sp_off_a","sp_def_h","elo_a","talent_a","ret_off_a","ret_def_h","portal_net_a","neutral_site","week"]
BASE_FEATS = ["sp_diff","sp_off_diff","sp_def_diff","elo_diff","talent_diff","same_hc_diff",
              "ret_off_h","ret_def_h","ret_off_a","ret_def_a","portal_net_h","portal_net_a",
              "ats_pct_diff","neutral_site","home_field","week"]

HGB_R = dict(max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=10, random_state=42)
HGB_C = dict(max_iter=300, max_depth=4, learning_rate=0.05, min_samples_leaf=10, random_state=42)
UNIT  = 100


def prep(df, feats):
    avail = [c for c in feats if c in df.columns]
    X = df[avail].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.reset_index(drop=True), avail


def payout(win, juice=-110):
    if win:
        return UNIT * (100 / abs(juice)) if juice < 0 else UNIT * (juice / 100)
    return -UNIT


def run():
    print("=" * 55)
    print("CFB ALGO v2 — Walk-Forward Backtest")
    print("Train on prior season, test on next season")
    print("=" * 55)

    path = os.path.join(DATA_DIR, "features_all.csv")
    if not os.path.exists(path):
        print("! features_all.csv not found"); return

    df = pd.read_csv(path, low_memory=False)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"]   = pd.to_numeric(df["week"],   errors="coerce")
    df = df.sort_values(["season","week"]).reset_index(drop=True)

    seasons = sorted(df["season"].dropna().unique())
    print(f"  Seasons available: {seasons}")
    print(f"  Total games: {len(df)}")

    all_results = []

    for i, test_season in enumerate(seasons):
        if i == 0:
            print(f"\n  Skipping {int(test_season)} — need prior season for training")
            continue

        # Train ONLY on prior season — strict temporal separation
        train_df = df[df["season"] == seasons[i-1]].copy()
        test_df  = df[df["season"] == test_season].copy()

        # Only train on games with actual scores
        # Regular season only — bowl games from training year overlap with test year start
        train_df = train_df[train_df["seasonType"] == "regular"].copy() if "seasonType" in train_df.columns else train_df
        train_scores = train_df.dropna(subset=["home_score","away_score"]).reset_index(drop=True)
        train_spread = train_df.dropna(subset=["cover_margin"]).reset_index(drop=True) if "cover_margin" in train_df.columns else pd.DataFrame()

        print(f"\n  Season {int(test_season)}")
        print(f"    Train: {int(seasons[i-1])} — {len(train_scores)} scored games, {len(train_spread)} with spread")
        print(f"    Test:  {int(test_season)} — {len(test_df)} games")

        if len(train_scores) < 100:
            print(f"    ! Not enough training data — skipping")
            continue

        # Train home score model
        X_home, hf = prep(train_scores, HOME_FEATS)
        home_model = HistGradientBoostingRegressor(**HGB_R)
        home_model.fit(X_home, train_scores["home_score"].values)

        # Train away score model
        X_away, af = prep(train_scores, AWAY_FEATS)
        away_model = HistGradientBoostingRegressor(**HGB_R)
        away_model.fit(X_away, train_scores["away_score"].values)

        # Train ML model
        ml_model, mf = None, []
        if "home_win" in train_scores.columns:
            ml_train = train_scores.dropna(subset=["home_win"]).reset_index(drop=True)
            X_ml, mf = prep(ml_train, BASE_FEATS)
            ml_model = HistGradientBoostingClassifier(**HGB_C)
            ml_model.fit(X_ml, ml_train["home_win"].values)

        # Test week by week
        season_res = []
        test_weeks = sorted(test_df["week"].dropna().unique())

        for week in test_weeks:
            wk = test_df[test_df["week"] == week].copy().reset_index(drop=True)

            X_h, _ = prep(wk, hf)
            X_a, _ = prep(wk, af)
            wk["pred_home"]   = home_model.predict(X_h)
            wk["pred_away"]   = away_model.predict(X_a)
            wk["pred_margin"] = wk["pred_home"] - wk["pred_away"]
            wk["pred_total"]  = wk["pred_home"] + wk["pred_away"]

            if ml_model and mf:
                X_m, _ = prep(wk, mf)
                wk["home_win_prob"] = ml_model.predict_proba(X_m)[:, 1]
            else:
                wk["home_win_prob"] = (0.5 + wk["pred_margin"] * 0.015).clip(0.05, 0.95)

            for _, row in wk.iterrows():
                if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
                    continue

                h_score  = float(row["home_score"])
                a_score  = float(row["away_score"])
                act_mar  = h_score - a_score
                act_tot  = h_score + a_score
                pred_mar = row["pred_margin"]
                pred_tot = row["pred_total"]
                hwp      = row["home_win_prob"]
                spread   = row.get("spread", np.nan)
                ou       = row.get("over_under", np.nan)
                hml      = row.get("home_moneyline", np.nan)
                aml      = row.get("away_moneyline", np.nan)

                res = {
                    "season": int(test_season), "week": int(week),
                    "home_team": row.get("home_team",""),
                    "away_team": row.get("away_team",""),
                    "pred_home": round(row["pred_home"],1),
                    "pred_away": round(row["pred_away"],1),
                    "actual_home": h_score, "actual_away": a_score,
                    "pred_margin": round(pred_mar,1), "actual_margin": act_mar,
                    "pred_total":  round(pred_tot,1), "actual_total":  act_tot,
                    "spread": spread, "over_under": ou,
                    "home_win_prob": round(hwp*100,1),
                }

                # Grade spread — only when line exists
                if not pd.isna(spread):
                    model_home = pred_mar + spread > 0
                    actual_home = act_mar + spread > 0
                    win = model_home == actual_home
                    res["spread_pick"]   = "home" if model_home else "away"
                    res["spread_result"] = "W" if win else "L"
                    res["spread_pnl"]    = payout(win)

                # Grade total — only when line exists
                if not pd.isna(ou):
                    model_over  = pred_tot > ou
                    actual_over = act_tot  > ou
                    win = model_over == actual_over
                    res["total_pick"]   = "over" if model_over else "under"
                    res["total_result"] = "W" if win else "L"
                    res["total_pnl"]    = payout(win)

                # Grade ML — only bet when model edge >= 5%
                if not pd.isna(hml) and not pd.isna(aml):
                    book_home_prob = (-float(hml)/(-float(hml)+100)) if float(hml)<0 else (100/(float(hml)+100))
                    edge = hwp - book_home_prob
                    if abs(edge) >= 0.05:
                        take_home = edge > 0
                        hw = act_mar > 0
                        win = take_home == hw
                        juice = float(hml) if take_home else float(aml)
                        res["ml_pick"]   = "home" if take_home else "away"
                        res["ml_result"] = "W" if win else "L"
                        res["ml_pnl"]    = payout(win, juice)
                        res["ml_edge"]   = round(edge*100,1)

                season_res.append(res)

        # Season summary
        sdf = pd.DataFrame(season_res)
        for market, col, pnl in [("Spread","spread_result","spread_pnl"),
                                  ("Total", "total_result", "total_pnl"),
                                  ("ML",    "ml_result",   "ml_pnl")]:
            sub = sdf.dropna(subset=[col]) if col in sdf.columns else pd.DataFrame()
            if sub.empty: continue
            w = (sub[col]=="W").sum()
            l = len(sub) - w
            p = sub[pnl].sum() if pnl in sub.columns else 0
            print(f"    {market}: {w}-{l} ({w/len(sub):.1%}) | PnL: ${p:.0f}")

        all_results.extend(season_res)

    # Overall
    all_df = pd.DataFrame(all_results)
    out = os.path.join(DATA_DIR, "backtest_results.csv")
    all_df.to_csv(out, index=False)

    print("\n" + "=" * 55)
    print("OVERALL BACKTEST (true out-of-sample)")
    print("=" * 55)
    for market, col, pnl in [("SPREAD","spread_result","spread_pnl"),
                              ("TOTAL", "total_result", "total_pnl"),
                              ("ML",    "ml_result",   "ml_pnl")]:
        sub = all_df.dropna(subset=[col]) if col in all_df.columns else pd.DataFrame()
        if sub.empty: print(f"  {market}: No data"); continue
        w = (sub[col]=="W").sum()
        l = len(sub) - w
        p = sub[pnl].sum() if pnl in sub.columns else 0
        roi = p / (len(sub) * UNIT) * 100
        print(f"  {market}: {w}-{l} ({w/len(sub):.1%}) | PnL: ${p:.0f} | ROI: {roi:.1f}%")

    print(f"\n  -> backtest_results.csv ({len(all_df)} graded games)")
    print("  Next: python predict.py 1")


if __name__ == "__main__":
    run()
