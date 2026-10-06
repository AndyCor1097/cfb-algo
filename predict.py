"""
predict.py — TheCFBAlgo Weekly Predictions
Uses CollegeFootballData weekly training CSV for opponent-adjusted metrics.

Usage:
  python predict.py 5          Generate Week 5 predictions
  python predict.py 5 --push   Generate + push to GitHub
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import subprocess
import requests
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
CFBD_KEY   = "b/c47r9EoNh7dOISJ+veiebAWqQIBphtZzdxz9UfdPG4b2GlaPmfKL9unFxI4+8R"
BASE       = "https://api.collegefootballdata.com"
HEADERS    = {"Authorization": f"Bearer {CFBD_KEY}", "Accept": "application/json"}
MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(MODEL_DIR, "data")
WEEK       = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PUSH       = "--push" in sys.argv

# ─── Team name normalization ───────────────────────────────────────────────────
TEAM_NORM = {
    # Common aliases → CFBD name
    "Ole Miss": "Mississippi",
    "USC": "Southern California",
    "LSU": "Louisiana State",
    "TCU": "Texas Christian",
    "UCF": "Central Florida",
    "UAB": "Alabama Birmingham",
    "UNLV": "Nevada Las Vegas",
    "SMU": "Southern Methodist",
    "BYU": "Brigham Young",
    "Pitt": "Pittsburgh",
    "Miami (OH)": "Miami Ohio",
    "UL Monroe": "Louisiana Monroe",
    "NC State": "North Carolina State",
    "UConn": "Connecticut",
    "USF": "South Florida",
    "FIU": "Florida International",
    "FAU": "Florida Atlantic",
    "WKU": "Western Kentucky",
    "NIU": "Northern Illinois",
    "UMass": "Massachusetts",
    "App State": "Appalachian State",
    "Sam Houston St": "Sam Houston State",
    "Jax State": "Jacksonville State",
    "E Michigan": "Eastern Michigan",
    "Sacramento St": "Sacramento State",
    "UT San Antonio": "UTSA",
    "Miami": "Miami",
    "Miami FL": "Miami",
}

def norm_team(name):
    if not name:
        return ""
    name = str(name).strip()
    return TEAM_NORM.get(name, name)


# ─── Home Field Advantage ──────────────────────────────────────────────────────
def load_hfa():
    hfa_path = os.path.join(DATA_DIR, "home_field_advantage.csv")
    if os.path.exists(hfa_path):
        hfa_df = pd.read_csv(hfa_path)
        if "team" in hfa_df.columns and "hfa_points" in hfa_df.columns:
            return dict(zip(hfa_df["team"], hfa_df["hfa_points"]))
    # Fallback defaults
    return {
        "Alabama": 3.5, "Ohio State": 3.5, "Georgia": 3.5,
        "LSU": 3.8, "Auburn": 3.8, "Penn State": 3.5,
        "Michigan": 3.2, "Texas": 3.2, "Tennessee": 3.2,
        "Clemson": 3.2, "Florida State": 3.0, "Notre Dame": 3.0,
        "Oregon": 3.0, "Washington": 2.8, "Wisconsin": 2.8,
    }

HFA = load_hfa()

def get_hfa(team, is_neutral):
    if is_neutral:
        return 0.0
    return HFA.get(norm_team(team), 2.2)


# ─── Lines from CSV ────────────────────────────────────────────────────────────
def load_lines():
    lines_path = os.path.join(DATA_DIR, "lines_2026.csv")
    if not os.path.exists(lines_path):
        return {}
    df = pd.read_csv(lines_path)
    lookup = {}
    for _, row in df.iterrows():
        if row.get("week") != WEEK:
            continue
        h = norm_team(str(row.get("home_team", "")))
        a = norm_team(str(row.get("away_team", "")))
        key = f"{h}|{a}"
        lookup[key] = {
            "spread":        row.get("spread"),
            "total":         row.get("over_under"),
            "home_ml":       row.get("home_moneyline"),
            "away_ml":       row.get("away_moneyline"),
            "commence_time": row.get("commence_time", ""),
        }
    return lookup


# ─── Core ratings margin from CSV ─────────────────────────────────────────────
def ratings_margin_from_row(row):
    """
    Calculate predicted point margin (home - away) from CSV metrics.
    Positive = home team favored.
    """

    # Adjusted EPA differential (offense minus opponent's defense allowed)
    # Higher EPA = better offense; lower EPA_allowed = better defense
    home_off_epa = float(row.get("home_adjusted_epa", 0) or 0)
    home_def_epa = float(row.get("home_adjusted_epa_allowed", 0) or 0)  # lower = better D
    away_off_epa = float(row.get("away_adjusted_epa", 0) or 0)
    away_def_epa = float(row.get("away_adjusted_epa_allowed", 0) or 0)

    # Net EPA: home offense - away defense allowed, and vice versa
    home_net_epa = home_off_epa - away_def_epa
    away_net_epa = away_off_epa - home_def_epa
    epa_diff = home_net_epa - away_net_epa  # positive favors home

    # Success rate differential
    home_sr_off = float(row.get("home_adjusted_success", 0) or 0)
    home_sr_def = float(row.get("home_adjusted_success_allowed", 0) or 0)
    away_sr_off = float(row.get("away_adjusted_success", 0) or 0)
    away_sr_def = float(row.get("away_adjusted_success_allowed", 0) or 0)

    home_sr_net = home_sr_off - away_sr_def
    away_sr_net = away_sr_off - home_sr_def
    sr_diff = home_sr_net - away_sr_net

    # Explosiveness differential
    home_exp = float(row.get("home_adjusted_explosiveness", 0) or 0)
    home_exp_def = float(row.get("home_adjusted_explosiveness_allowed", 0) or 0)
    away_exp = float(row.get("away_adjusted_explosiveness", 0) or 0)
    away_exp_def = float(row.get("away_adjusted_explosiveness_allowed", 0) or 0)

    home_exp_net = home_exp - away_exp_def
    away_exp_net = away_exp - home_exp_def
    exp_diff = home_exp_net - away_exp_net

    # Havoc (defense metric — higher havoc defense = better)
    home_havoc_d = float(row.get("home_total_havoc_defense", 0) or 0)
    away_havoc_d = float(row.get("away_total_havoc_defense", 0) or 0)
    havoc_diff = home_havoc_d - away_havoc_d  # positive = home has better D havoc

    # Points per opportunity differential
    home_ppo_off = float(row.get("home_points_per_opportunity_offense", 0) or 0)
    home_ppo_def = float(row.get("home_points_per_opportunity_defense", 0) or 0)
    away_ppo_off = float(row.get("away_points_per_opportunity_offense", 0) or 0)
    away_ppo_def = float(row.get("away_points_per_opportunity_defense", 0) or 0)

    home_ppo_net = home_ppo_off - away_ppo_def
    away_ppo_net = away_ppo_off - home_ppo_def
    ppo_diff = home_ppo_net - away_ppo_net

    # Elo differential
    home_elo = float(row.get("home_elo", 1500) or 1500)
    away_elo = float(row.get("away_elo", 1500) or 1500)
    elo_diff = home_elo - away_elo  # e.g. 100 Elo = roughly 2.8 pts

    # Talent differential
    home_talent = float(row.get("home_talent", 800) or 800)
    away_talent = float(row.get("away_talent", 800) or 800)
    talent_diff = home_talent - away_talent

    # ─── Weighted composite margin ─────────────────────────────────────────
    # Scales calibrated via regression against 2026 in-season spreads
    # EPA range ~0.28, Elo range ~1000, success rate range ~0.15

    # Coefficients from regression against 2026 in-season spreads (MAE: 4.6 pts)
    margin = (
        elo_diff    * 0.0341 +   # Elo (regression-calibrated)
        epa_diff    * 16.65  +   # Opponent-adjusted EPA (regression-calibrated)
        sr_diff     * -5.01  +   # Success rate (negative = allowed hurts more)
        ppo_diff    * -0.15  +   # Points per opportunity
        2.40                     # Intercept (slight home field baseline)
    )

    return margin


# ─── Win probability ───────────────────────────────────────────────────────────
def margin_to_wp(margin):
    """Convert predicted margin to home win probability."""
    # Sigmoid: roughly 3.5 pt margin = 60% WP
    return round(100 / (1 + np.exp(-margin / 7.5)), 1)


# ─── Moneyline from WP ────────────────────────────────────────────────────────
def wp_to_ml(wp):
    """Convert win probability to American moneyline."""
    if wp is None:
        return None, None
    p = wp / 100.0
    p = max(0.01, min(0.99, p))
    if p >= 0.5:
        fav_ml = int(round(-p / (1 - p) * 100))
        dog_ml = int(round((1 - p) / p * 100))
    else:
        fav_ml = int(round(p / (1 - p) * 100))
        dog_ml = int(round(-(1 - p) / p * 100))
    home_ml = fav_ml if p >= 0.5 else dog_ml
    away_ml = dog_ml if p >= 0.5 else fav_ml
    return home_ml, away_ml


# ─── Spread edge label ────────────────────────────────────────────────────────
def edge_label(model_margin, book_spread, home_team, away_team):
    """
    book_spread is from home team's perspective (negative = home favored).
    model_margin is positive when home team is predicted to win.
    """
    if book_spread is None:
        return None, None, None
    # Convert book spread to model comparison
    # If home is -7, book expects home by 7. If model says +10, edge is +3 for home.
    model_vs_book = model_margin - (-float(book_spread))  # positive = home better than book
    edge_pts = abs(model_vs_book)
    if edge_pts < 2:
        return None, None, None
    if model_vs_book > 0:
        pick_team = home_team
        pick_side = "home"
    else:
        pick_team = away_team
        pick_side = "away"
    if edge_pts >= 7:
        label = "🔥 STRONG"
    elif edge_pts >= 4:
        label = "✅ EDGE"
    else:
        label = "📊 LEAN"
    return label, pick_team, pick_side


# ─── Main prediction loop ──────────────────────────────────────────────────────
def run():
    print("=" * 55)
    print(f"  TheCFBAlgo — Week {WEEK} Predictions  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 55)

    # Load CSV
    csv_path = os.path.join(DATA_DIR, f"training_data_2026_week{WEEK:02d}.csv")
    if not os.path.exists(csv_path):
        # Try without zero padding
        csv_path = os.path.join(DATA_DIR, f"training_data_2026_week{WEEK}.csv")
    if not os.path.exists(csv_path):
        print(f"  ! CSV not found: {csv_path}")
        print(f"    Drop your training_data_2026_week{WEEK:02d}.csv in data/ folder")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} games from CSV")

    # Filter to this week (CSV may have multiple weeks)
    if "week" in df.columns:
        week_df = df[df["week"] == WEEK].copy()
        if len(week_df) == 0:
            week_df = df.copy()  # Use all rows if week filter returns nothing
        print(f"  Week {WEEK} games: {len(week_df)}")
    else:
        week_df = df.copy()

    # Load lines and HFA
    lines = load_lines()
    print(f"  Lines loaded: {len(lines)} matchups")

    games = []
    for _, row in week_df.iterrows():
        home = norm_team(str(row.get("home_team", "")))
        away = norm_team(str(row.get("away_team", "")))
        if not home or not away:
            continue

        neutral = str(row.get("neutral_site", "FALSE")).upper() == "TRUE"
        start_date = str(row.get("start_date", ""))

        # Get book lines
        line_key = f"{home}|{away}"
        line_data = lines.get(line_key, {})

        # Also check if CSV has a spread
        csv_spread = row.get("spread")
        book_spread = line_data.get("spread")
        if (book_spread is None or pd.isna(book_spread)) and csv_spread is not None and not pd.isna(csv_spread):
            book_spread = float(csv_spread)

        book_total = line_data.get("total")
        home_ml_book = line_data.get("home_ml")
        away_ml_book = line_data.get("away_ml")

        # Convert types
        book_spread = float(book_spread) if book_spread is not None and not pd.isna(book_spread) else None
        book_total = float(book_total) if book_total is not None and not pd.isna(book_total) else None

        # Model margin (positive = home wins)
        ratings_margin = ratings_margin_from_row(row)

        # Home field advantage
        hfa = get_hfa(home, neutral)
        ratings_margin += hfa

        # ─── Market blend ──────────────────────────────────────────────────
        if book_spread is not None:
            # book_spread convention: negative = home favored
            market_margin = -book_spread  # convert to home margin

            abs_diff = abs(ratings_margin - market_margin)
            if abs_diff > 25:
                w_model, w_market = 0.95, 0.05
            elif abs_diff > 15:
                w_model, w_market = 0.90, 0.10
            else:
                w_model, w_market = 0.85, 0.15

            final_margin = w_model * ratings_margin + w_market * market_margin
        else:
            final_margin = ratings_margin

        # ─── Scores ────────────────────────────────────────────────────────
        # Use book total if available, else estimate from Elo/context
        home_elo = float(row.get("home_elo", 1500) or 1500)
        away_elo = float(row.get("away_elo", 1500) or 1500)

        if book_total is not None:
            total = book_total
        else:
            # Estimate total from team quality (Elo-based)
            avg_elo = (home_elo + away_elo) / 2
            # ~1500 Elo = ~48 pts, ±100 Elo = ±1.5 pts per team
            total = 48 + (avg_elo - 1500) * 0.015

        home_score = round((total + final_margin) / 2)
        away_score = round((total - final_margin) / 2)

        # Ensure no ties
        if home_score == away_score:
            home_score += 1

        # Floor scores at 7 to avoid unrealistic numbers
        home_score = max(7, home_score)
        away_score = max(7, away_score)

        pred_total = home_score + away_score

        # ─── Win probability & ML ──────────────────────────────────────────
        home_wp = margin_to_wp(final_margin)
        home_ml, away_ml = wp_to_ml(home_wp)

        # ─── Spread edge ───────────────────────────────────────────────────
        edge_lbl, edge_pick, edge_side = edge_label(final_margin, book_spread, home, away)

        # Spread edge value (positive = model likes home vs book)
        if book_spread is not None:
            spread_edge = final_margin - (-book_spread)
        else:
            spread_edge = None

        # Total edge
        if book_total is not None:
            total_edge = pred_total - book_total
        else:
            total_edge = None

        # ML edge
        if home_ml_book and home_ml:
            try:
                book_prob = abs(float(home_ml_book)) / (abs(float(home_ml_book)) + 100) if float(home_ml_book) < 0 else 100 / (float(home_ml_book) + 100)
                ml_edge = home_wp / 100 - book_prob
            except:
                ml_edge = None
        else:
            ml_edge = None

        game = {
            "week":            WEEK,
            "home_team":       home,
            "away_team":       away,
            "start_date":      start_date,
            "neutral_site":    neutral,
            "pred_home":       home_score,
            "pred_away":       away_score,
            "pred_margin":     round(final_margin, 1),
            "pred_total":      pred_total,
            "ratings_margin":  round(ratings_margin, 1),
            "home_win_prob":   home_wp,
            "home_field_adv":  hfa,
            "book_spread":     book_spread,
            "book_total":      book_total,
            "home_moneyline":  home_ml,
            "away_moneyline":  away_ml,
            "home_moneyline_book": home_ml_book,
            "away_moneyline_book": away_ml_book,
            "spread_edge":     round(spread_edge, 1) if spread_edge is not None else None,
            "total_edge":      round(total_edge, 1) if total_edge is not None else None,
            "ml_edge":         round(ml_edge, 3) if ml_edge is not None else None,
            "edge_label":      edge_lbl,
            "edge_pick":       edge_pick,
            "edge_side":       edge_side,
            # Elo for display
            "home_elo":        int(home_elo),
            "away_elo":        int(away_elo),
            # Kickoff time from Odds API
            "commence_time":   line_data.get("commence_time", ""),
        }
        games.append(game)

    # Sort by start date then home team
    games.sort(key=lambda g: (g.get("start_date", ""), g.get("home_team", "")))

    # ─── Summary ─────────────────────────────────────────────────────────────
    print(f"\n  {len(games)} predictions generated\n")
    print(f"  {'HOME':<22} {'AWAY':<22} {'SCORE':>12}  MARGIN")
    print(f"  {'-'*22} {'-'*22} {'-'*12}  {'------'}")
    for g in games:
        score_str = f"{g['pred_home']}-{g['pred_away']}"
        margin = g['pred_margin']
        fav = g['home_team'] if margin > 0 else g['away_team']
        edge = f" [{g['edge_label']}]" if g.get("edge_label") else ""
        print(f"  {g['home_team']:<22} {g['away_team']:<22} {score_str:>12}  {margin:+.1f}{edge}")

    # ─── Save JSON ────────────────────────────────────────────────────────────
    out_path = os.path.join(MODEL_DIR, f"predictions_week{WEEK}.json")
    out = {
        "week":       WEEK,
        "generated":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model":      "CSV-based opponent-adjusted metrics (85% ratings / 15% market)",
        "games":      games,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> {out_path} saved ({len(games)} games)")

    # ─── Push to GitHub ───────────────────────────────────────────────────────
    if PUSH:
        print("\n  Pushing to GitHub...")
        cmds = [
            ["git", "add", f"predictions_week{WEEK}.json"],
            ["git", "commit", "-m", f"Week {WEEK} predictions ({len(games)} games)"],
            ["git", "push"],
        ]
        for cmd in cmds:
            result = subprocess.run(cmd, cwd=MODEL_DIR, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ! {' '.join(cmd)}: {result.stderr.strip()}")
            else:
                print(f"  ✓ {' '.join(cmd)}")
        print(f"  Site: https://thecfbalgo.streamlit.app")

    print("\n  Done!")


if __name__ == "__main__":
    run()
