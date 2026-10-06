"""
model.py — Train on 2023-2025 with full feature set including PPA
"""

import pandas as pd, numpy as np, os, joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

HOME_SCORE_FEATURES = [
    "sp_off_h", "sp_def_a",
    "elo_h",
    "ppa_off_h", "ppa_def_a",
    "adv_off_ppa_h", "adv_def_ppa_a",
    "adv_off_sr_h", "adv_def_sr_a",
    "adv_off_exp_h", "adv_def_exp_a",
    "adv_havoc_a",
    "talent_h", "ret_off_h", "ret_def_a",
    "portal_net_h",
    "neutral_site", "home_field", "week",
]

AWAY_SCORE_FEATURES = [
    "sp_off_a", "sp_def_h",
    "elo_a",
    "ppa_off_a", "ppa_def_h",
    "adv_off_ppa_a", "adv_def_ppa_h",
    "adv_off_sr_a", "adv_def_sr_h",
    "adv_off_exp_a", "adv_def_exp_h",
    "adv_havoc_h",
    "talent_a", "ret_off_a", "ret_def_h",
    "portal_net_a",
    "neutral_site", "week",
]

BASE_FEATURES = [
    "sp_diff", "sp_off_diff", "sp_def_diff",
    "elo_diff",
    "ppa_off_diff", "ppa_def_diff",
    "adv_off_ppa_diff", "adv_def_ppa_diff",
    "adv_off_sr_diff", "adv_def_sr_diff",
    "adv_off_exp_diff", "adv_def_exp_diff",
    "adv_havoc_diff",
    "talent_diff", "same_hc_diff",
    "ret_off_h", "ret_def_h", "ret_off_a", "ret_def_a",
    "portal_net_h", "portal_net_a",
    "ats_pct_diff",
    "neutral_site", "home_field", "week",
]

HGB = dict(max_iter=400, max_depth=5, learning_rate=0.04,
           min_samples_leaf=8, random_state=42, l2_regularization=0.1)


def prep(df, feats):
    avail = [c for c in feats if c in df.columns]
    X = df[avail].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.reset_index(drop=True), avail


def cv_eval(X, y, label):
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    print(f"\n  {label}:")
    print(f"  {'Fold':<5} {'MAE':<8} N")
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        m = HistGradientBoostingRegressor(**HGB)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        mae = mean_absolute_error(y.iloc[te], p)
        maes.append(mae)
        print(f"  {i:<5} {mae:<8.2f} {len(te)}")
    print(f"  Avg MAE: {np.mean(maes):.2f} points")
    return np.mean(maes)


def train():
    print("="*50)
    print("CFB ALGO — Model Training")
    print("Full feature set: SP+, Elo, PPA, Advanced Stats")
    print("="*50)

    path = os.path.join(DATA_DIR, "features_all.csv")
    if not os.path.exists(path):
        print("  ! features_all.csv not found"); return

    df = pd.read_csv(path, low_memory=False).sort_values(["season","week"]).reset_index(drop=True)
    print(f"  {len(df)} total games")

    score_df = df.dropna(subset=["home_score","away_score"]).copy() if "home_score" in df.columns else pd.DataFrame()
    if score_df.empty:
        print("  ! No score data"); return
    score_df = score_df.reset_index(drop=True)
    print(f"  {len(score_df)} with scores")

    # Show available features
    ppa_avail = [c for c in score_df.columns if "ppa" in c.lower() or "adv" in c.lower()]
    print(f"  PPA/Advanced features available: {len(ppa_avail)}")

    # Home score model
    X_home, home_feats = prep(score_df, HOME_SCORE_FEATURES)
    y_home = score_df["home_score"].reset_index(drop=True)
    home_mae = cv_eval(X_home, y_home, "Home Score Model")
    home_model = HistGradientBoostingRegressor(**HGB)
    home_model.fit(X_home, y_home)
    print(f"  Home features used: {home_feats}")

    # Away score model
    X_away, away_feats = prep(score_df, AWAY_SCORE_FEATURES)
    y_away = score_df["away_score"].reset_index(drop=True)
    away_mae = cv_eval(X_away, y_away, "Away Score Model")
    away_model = HistGradientBoostingRegressor(**HGB)
    away_model.fit(X_away, y_away)

    # Validation
    home_preds = home_model.predict(X_home)
    away_preds = away_model.predict(X_away)
    ats_acc, ou_acc = 0, 0

    if "spread" in score_df.columns and "actual_margin" in score_df.columns:
        mask = score_df["spread"].notna() & score_df["actual_margin"].notna()
        rows = score_df[mask]
        if not rows.empty:
            idx = rows.index.tolist()
            pm  = (home_preds - away_preds)[score_df.index.get_indexer(idx)]
            ats_acc = np.mean((pm + rows["spread"].values > 0) == (rows["actual_margin"].values + rows["spread"].values > 0))
            print(f"\n  ATS (in-sample): {ats_acc:.1%}")

    if "over_under" in score_df.columns and "actual_total" in score_df.columns:
        mask2 = score_df["over_under"].notna() & score_df["actual_total"].notna()
        rows2 = score_df[mask2]
        if not rows2.empty:
            idx2 = rows2.index.tolist()
            pt   = (home_preds + away_preds)[score_df.index.get_indexer(idx2)]
            ou_acc = np.mean((pt > rows2["over_under"].values) == (rows2["actual_total"].values > rows2["over_under"].values))
            print(f"  O/U (in-sample): {ou_acc:.1%}")

    # Direct spread model
    spread_df = df.dropna(subset=["cover_margin"]).copy() if "cover_margin" in df.columns else pd.DataFrame()
    spread_model, spread_feats = None, []
    if not spread_df.empty:
        spread_df = spread_df.reset_index(drop=True)
        X_sp, sp_feats = prep(spread_df, BASE_FEATURES)
        y_sp = spread_df["cover_margin"].reset_index(drop=True)
        cv_eval(X_sp, y_sp, "Direct Spread Model")
        spread_model = HistGradientBoostingRegressor(**HGB)
        spread_model.fit(X_sp, y_sp)
        spread_feats = sp_feats

    # ML model
    ml_model, ml_feats = None, []
    ml_df = df.dropna(subset=["home_win"]).copy() if "home_win" in df.columns else pd.DataFrame()
    if not ml_df.empty:
        ml_df = ml_df.reset_index(drop=True)
        X_ml, ml_f = prep(ml_df, BASE_FEATURES)
        y_ml = ml_df["home_win"].reset_index(drop=True)
        ml_clf = HistGradientBoostingClassifier(
            max_iter=400, max_depth=4, learning_rate=0.04,
            min_samples_leaf=10, random_state=42
        )
        ml_clf.fit(X_ml, y_ml)
        ml_model, ml_feats = ml_clf, ml_f

    joblib.dump({
        "home_model":   home_model,   "home_feats":   home_feats,   "home_mae":   home_mae,
        "away_model":   away_model,   "away_feats":   away_feats,   "away_mae":   away_mae,
        "spread_model": spread_model, "spread_feats": spread_feats,
        "ml_model":     ml_model,     "ml_feats":     ml_feats,
        "ats_acc":      ats_acc,      "ou_acc":       ou_acc,
        "trained_on":   "2023-2025",
    }, os.path.join(MODEL_DIR, "cfb_model.pkl"))

    print(f"\n  -> cfb_model.pkl saved")
    print(f"  Home MAE: {home_mae:.2f} | Away MAE: {away_mae:.2f}")
    print(f"  ATS: {ats_acc:.1%} | O/U: {ou_acc:.1%}")
    print("  Next: python predict.py 4 --push")


if __name__ == "__main__":
    train()
