"""
predict_fcs.py — Generate predictions for FBS vs FCS games
Saves predictions_fcs_week{N}.json for the app's FCS tab

Usage:
  python predict_fcs.py 1    Week 1
"""

import pandas as pd, numpy as np, os, joblib, sys, json
from datetime import datetime

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
WEEK_FILTER = int(sys.argv[1]) if len(sys.argv) > 1 else None

FBS_CONFERENCES = [
    "ACC","Big 12","Big Ten","Pac-12","SEC",
    "American Athletic","Conference USA","MAC",
    "Mountain West","Sun Belt","FBS Independents","American","Ind"
]


def load_model():
    path = os.path.join(MODEL_DIR, "cfb_model.pkl")
    if not os.path.exists(path): return None
    return joblib.load(path)


def prep(df, feats):
    avail = [c for c in feats if c in df.columns]
    X = df[avail].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def ml_to_impl(ml):
    try:
        ml = float(ml)
        return round((-ml/(-ml+100))*100 if ml<0 else (100/(ml+100))*100, 1)
    except: return None


def run():
    print("=" * 50)
    print(f"TheCFBAlgo — FCS Predictions  |  {datetime.now():%Y-%m-%d %H:%M}")
    if WEEK_FILTER: print(f"Week: {WEEK_FILTER}")
    print("=" * 50)

    bundle = load_model()
    if not bundle:
        print("  ! cfb_model.pkl not found"); return

    # Load ALL 2026 games (not FBS filtered)
    all_games_path = os.path.join(DATA_DIR, "schedule_2026.csv")
    if not os.path.exists(all_games_path):
        print("  ! schedule_2026.csv not found"); return

    games = pd.read_csv(all_games_path, low_memory=False)
    games = games.rename(columns={
        "homeTeam": "home_team", "awayTeam": "away_team",
        "neutralSite": "neutral_site", "id": "game_id",
    })

    if WEEK_FILTER and "week" in games.columns:
        games = games[games["week"] == WEEK_FILTER].copy()

    # Keep only FBS vs FCS games (home is FBS, away is FCS or vice versa)
    if "homeClassification" in games.columns:
        fbs_home_fcs_away = (games["homeClassification"] == "fbs") & (games["awayClassification"] != "fbs")
        fcs_home_fbs_away = (games["homeClassification"] != "fbs") & (games["awayClassification"] == "fbs")
        games = games[fbs_home_fcs_away | fcs_home_fbs_away].copy()
    elif "homeConference" in games.columns:
        home_fbs = games["homeConference"].isin(FBS_CONFERENCES)
        away_fbs = games["awayConference"].isin(FBS_CONFERENCES)
        games = games[(home_fbs & ~away_fbs) | (~home_fbs & away_fbs)].copy()

    print(f"  {len(games)} FBS vs FCS games found")
    if games.empty: return

    # Load features for the FBS teams and merge
    feat_path = os.path.join(DATA_DIR, "features_2026.csv")
    features_fbs = pd.read_csv(feat_path, low_memory=False) if os.path.exists(feat_path) else pd.DataFrame()

    # For FCS games we only have ratings for the FBS side
    # Fill FCS team features with league-average FCS values
    FCS_DEFAULTS = {
        "sp_h": -15, "sp_off_h": -8, "sp_def_h": -8,
        "sp_a": -15, "sp_off_a": -8, "sp_def_a": -8,
        "elo_h": 1350, "elo_a": 1350,
        "talent_h": 600, "talent_a": 600,
        "ret_off_h": 0.5, "ret_off_a": 0.5,
        "ret_def_h": 0.5, "ret_def_a": 0.5,
        "portal_net_h": 0, "portal_net_a": 0,
        "neutral_site": 0, "home_field": 1, "week": WEEK_FILTER or 1,
    }

    # Build a feature row per game using FBS team's actual features where possible
    rows = []
    for _, game in games.iterrows():
        home = game.get("home_team","")
        away = game.get("away_team","")
        week = game.get("week", WEEK_FILTER or 1)

        # Try to find FBS team features
        fbs_row = pd.Series(FCS_DEFAULTS)
        if not features_fbs.empty:
            home_feat = features_fbs[features_fbs["home_team"] == home]
            if not home_feat.empty:
                fbs_row = home_feat.iloc[0]

        row = dict(fbs_row)
        row["home_team"] = home
        row["away_team"] = away
        row["week"]      = week
        row["game_id"]   = game.get("game_id", game.get("id",""))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill missing model features
    all_feats = list(set(
        bundle.get("home_feats",[]) + bundle.get("away_feats",[]) +
        bundle.get("spread_feats",[]) + bundle.get("ml_feats",[])
    ))
    for col in all_feats:
        if col not in df.columns:
            df[col] = 0.0

    # Predict
    X_h = prep(df, bundle["home_feats"])
    X_a = prep(df, bundle["away_feats"])
    df["pred_home"]   = np.round(bundle["home_model"].predict(X_h), 1)
    df["pred_away"]   = np.round(bundle["away_model"].predict(X_a), 1)
    df["pred_margin"] = (df["pred_home"] - df["pred_away"]).round(1)
    df["pred_total"]  = (df["pred_home"] + df["pred_away"]).round(1)
    df["home_win_prob"] = (50 + df["pred_margin"]*1.5).clip(5,99).round(1)
    df["away_win_prob"] = (100 - df["home_win_prob"]).round(1)

    # Merge lines
    lines_path = os.path.join(DATA_DIR, "lines_2026.csv")
    if os.path.exists(lines_path):
        lines = pd.read_csv(lines_path, low_memory=False)
        lines["_p"] = lines["provider"].str.lower().str.strip()
        best = pd.DataFrame()
        for p in ["fanduel","draftkings","espn bet","bovada","consensus"]:
            sub = lines[lines["_p"]==p]
            if not sub.empty:
                best = sub.groupby(["home_team","away_team"]).first().reset_index()
                break
        if not best.empty:
            for col in ["spread","over_under","home_moneyline","away_moneyline"]:
                if col in df.columns: df.drop(columns=[col], inplace=True)
            df = df.merge(
                best[["home_team","away_team","spread","over_under","home_moneyline","away_moneyline"]],
                on=["home_team","away_team"], how="left"
            )

    for col in ["spread","over_under","home_moneyline","away_moneyline"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["model_spread"] = -df["pred_margin"]
    df["spread_edge"]  = (df["spread"] - df["model_spread"]).round(1)
    df["total_edge"]   = (df["pred_total"] - df["over_under"]).round(1)
    df["book_home_impl"] = df["home_moneyline"].apply(ml_to_impl)
    df["ml_home_edge"] = (df["home_win_prob"] - df["book_home_impl"].fillna(50)).round(1)

    # Only games with lines
    df = df[df["spread"].notna()].copy() if "spread" in df.columns else df
    print(f"  {len(df)} FCS games with lines")

    # Build output
    def safe(v, r=1):
        try:
            f = float(v)
            return round(f,r) if not np.isnan(f) else None
        except: return None

    def safe_int(v):
        try:
            f = float(v)
            return int(round(f)) if not np.isnan(f) else None
        except: return None

    game_list = []
    for _, row in df.iterrows():
        game_list.append({
            "week":           safe_int(row.get("week")),
            "home_team":      str(row.get("home_team","")),
            "away_team":      str(row.get("away_team","")),
            "pred_home":      safe_int(row.get("pred_home")),
            "pred_away":      safe_int(row.get("pred_away")),
            "pred_margin":    safe(row.get("pred_margin")),
            "pred_total":     safe(row.get("pred_total")),
            "home_win_prob":  safe(row.get("home_win_prob")),
            "away_win_prob":  safe(row.get("away_win_prob")),
            "book_spread":    safe(row.get("spread")),
            "model_spread":   safe(row.get("model_spread")),
            "spread_edge":    safe(row.get("spread_edge")),
            "book_total":     safe(row.get("over_under")),
            "total_edge":     safe(row.get("total_edge")),
            "home_moneyline": safe_int(row.get("home_moneyline")),
            "away_moneyline": safe_int(row.get("away_moneyline")),
            "ml_home_edge":   safe(row.get("ml_home_edge")),
        })

    game_list.sort(key=lambda x: abs(x.get("spread_edge") or 0), reverse=True)

    out = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "week":         WEEK_FILTER,
        "game_count":   len(game_list),
        "note":         "FBS vs FCS games — model trained on FBS data only, use with caution",
        "games":        game_list,
    }

    week_str = f"_week{WEEK_FILTER}" if WEEK_FILTER else "_all"
    json_path = os.path.join(MODEL_DIR, f"predictions_fcs{week_str}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> predictions_fcs{week_str}.json saved ({len(game_list)} games)")

    # Preview
    print(f"\n  {'Wk':<4} {'Home':<25} {'Pred':>4} {'Away':<25} {'Pred':>4} {'Sprd':>7} {'Edge':>6}")
    for g in game_list[:15]:
        sp = f"{g['book_spread']:+.1f}" if g['book_spread'] is not None else "N/A"
        se = f"{g['spread_edge']:+.1f}" if g['spread_edge'] is not None else "N/A"
        print(f"  {str(g['week']):<4} {g['home_team']:<25} {str(g['pred_home']):>4} "
              f"{g['away_team']:<25} {str(g['pred_away']):>4} {sp:>7} {se:>6}")


if __name__ == "__main__":
    run()
