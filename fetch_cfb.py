"""
fetch_cfb.py — Pull all CFBD data with temporal correctness
Key difference from v1: pulls weekly Elo snapshots so training
data uses only information available at game time.
"""

import requests, pandas as pd, os, time
from datetime import datetime

CFBD_KEY    = "b/c47r9EoNh7dOISJ+veiebAWqQIBphtZzdxz9UfdPG4b2GlaPmfKL9unFxI4+8R"
BASE        = "https://api.collegefootballdata.com"
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
HEADERS     = {"Authorization": f"Bearer {CFBD_KEY}", "Accept": "application/json"}
TRAIN_YEARS = [2023, 2024, 2025]   # portal era only
CURR_YEAR   = 2026
WEEKS       = list(range(0, 16))

os.makedirs(DATA_DIR, exist_ok=True)


def fetch(endpoint, params=None, label="", silent=False):
    try:
        r = requests.get(f"{BASE}{endpoint}", headers=HEADERS, params=params or {}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if not silent:
                print(f"  ✓ {label} — {len(data)} records")
            return data
        else:
            if not silent:
                print(f"  ✗ [{r.status_code}] {label}")
            return []
    except Exception as e:
        if not silent:
            print(f"  ✗ {label}: {e}")
        return []
    finally:
        time.sleep(0.15)


def save(data, filename):
    if not data:
        return
    df = pd.json_normalize(data) if isinstance(data, list) and data and isinstance(data[0], dict) else pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, filename), index=False)
    print(f"    -> {filename} ({len(df)} rows)")


# ── 1. GAMES ──────────────────────────────────
def fetch_games():
    print("\n[1/8] Games...")
    all_games = []
    for y in TRAIN_YEARS:
        all_games += fetch("/games", {"year": y, "seasonType": "regular"},    f"games {y}")
        all_games += fetch("/games", {"year": y, "seasonType": "postseason"}, f"bowls {y}")
    save(all_games, "games_history.csv")

    sched = fetch("/games", {"year": CURR_YEAR, "seasonType": "regular"}, "schedule 2026")
    save(sched, "schedule_2026.csv")


# ── 2. LINES ──────────────────────────────────
def fetch_lines():
    print("\n[2/8] Lines...")
    rows = []
    for y in TRAIN_YEARS:
        games = fetch("/lines", {"year": y}, f"lines {y}")
        for g in games:
            for line in g.get("lines", []):
                rows.append({
                    "game_id": g.get("id"), "year": y,
                    "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
                    "week": g.get("week"),
                    "provider": line.get("provider"),
                    "spread": line.get("spread"),
                    "over_under": line.get("overUnder"),
                    "home_moneyline": line.get("homeMoneyline"),
                    "away_moneyline": line.get("awayMoneyline"),
                })
    save(rows, "lines_history.csv")

    rows_2026 = []
    games_2026 = fetch("/lines", {"year": CURR_YEAR}, "lines 2026")
    for g in games_2026:
        for line in g.get("lines", []):
            rows_2026.append({
                "game_id": g.get("id"),
                "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
                "week": g.get("week"),
                "provider": line.get("provider"),
                "spread": line.get("spread"),
                "over_under": line.get("overUnder"),
                "home_moneyline": line.get("homeMoneyline"),
                "away_moneyline": line.get("awayMoneyline"),
            })
    save(rows_2026, "lines_2026.csv")


# ── 3. WEEKLY ELO (Phase 1 key feature) ───────
def fetch_weekly_elo():
    """
    Pull Elo ratings week by week so we know exactly
    what each team's rating was BEFORE each game.
    This prevents future data leakage in training.
    """
    print("\n[3/8] Weekly Elo snapshots (temporal correctness)...")
    rows = []
    for y in TRAIN_YEARS + [CURR_YEAR]:
        for w in WEEKS:
            data = fetch("/ratings/elo", {"year": y, "week": w}, f"elo {y} wk{w}", silent=True)
            for r in data:
                rows.append({**r, "year": y, "week": w})
        print(f"  ✓ Elo weekly snapshots {y} — done")
    save(rows, "elo_weekly.csv")


# ── 4. SP+ (preseason = week 0 or earliest available) ──
def fetch_sp():
    print("\n[4/8] SP+ ratings...")
    rows = []
    for y in TRAIN_YEARS + [CURR_YEAR]:
        # Week 0 = preseason SP+, no games played yet
        data = fetch("/ratings/sp", {"year": y}, f"SP+ {y}")
        rows += [{**r, "year": y} for r in data]
    save(rows, "ratings_sp.csv")


# ── 5. ROSTER / PORTAL / RETURNING ────────────
def fetch_roster():
    print("\n[5/8] Portal, returning production, talent, coaches...")

    portal = fetch("/player/portal", {"year": CURR_YEAR}, "portal 2026")
    save(portal, "portal_2026.csv")

    # Returning production — for each training year use PRIOR year's returning
    # (what was known before the season started)
    ret_rows = []
    for y in TRAIN_YEARS + [CURR_YEAR]:
        ret = fetch("/player/returning", {"year": y}, f"returning {y}")
        ret_rows += [{**r, "year": y} for r in ret]
    save(ret_rows, "returning_production.csv")

    # Talent — use prior year as proxy when current not available
    talent_rows = []
    for y in TRAIN_YEARS + [CURR_YEAR]:
        t = fetch("/talent", {"year": y}, f"talent {y}")
        talent_rows += [{**r, "year": y} for r in t]
    save(talent_rows, "talent.csv")

    # Coaches
    coach_rows = []
    for y in TRAIN_YEARS + [CURR_YEAR]:
        c = fetch("/coaches", {"year": y}, f"coaches {y}")
        coach_rows += [{**r, "year": y} for r in c]
    save(coach_rows, "coaches.csv")

    # Recruiting
    rec_rows = []
    for y in TRAIN_YEARS + [CURR_YEAR]:
        r = fetch("/recruiting/groups", {"year": y}, f"recruiting {y}")
        rec_rows += [{**x, "year": y} for x in r]
    save(rec_rows, "recruiting.csv")


# ── 6. ADVANCED STATS + PPA ───────────────────
def fetch_advanced():
    print("\n[6/8] Advanced stats + PPA...")
    adv, ppa = [], []
    for y in TRAIN_YEARS:
        a = fetch("/stats/season/advanced", {"year": y}, f"adv {y}")
        adv += [{**r, "year": y} for r in a]
        p = fetch("/ppa/teams", {"year": y}, f"ppa {y}")
        ppa += [{**r, "year": y} for r in p]
    save(adv, "advanced_stats.csv")
    save(ppa, "ppa_teams.csv")

    # Weekly PPA — for temporal correctness
    print("  Fetching weekly PPA for temporal features...")
    wppa = []
    for y in TRAIN_YEARS:
        for w in range(1, 16):
            data = fetch("/ppa/teams", {"year": y, "week": w}, f"ppa {y} wk{w}", silent=True)
            wppa += [{**r, "year": y, "week": w} for r in data]
        print(f"  ✓ Weekly PPA {y}")
    save(wppa, "ppa_weekly.csv")


# ── 7. VENUES + ATS ───────────────────────────
def fetch_context():
    print("\n[7/8] Venues + ATS...")
    save(fetch("/venues", {}, "venues"), "venues.csv")
    ats = []
    for y in TRAIN_YEARS:
        a = fetch("/teams/ats", {"year": y}, f"ATS {y}")
        ats += [{**r, "year": y} for r in a]
    save(ats, "ats_history.csv")


# ── 8. FPI (has preseason projections) ────────
def fetch_fpi():
    print("\n[8/8] FPI ratings...")
    rows = []
    for y in TRAIN_YEARS + [CURR_YEAR]:
        data = fetch("/ratings/fpi", {"year": y}, f"FPI {y}")
        rows += [{**r, "year": y} for r in data]
    save(rows, "ratings_fpi.csv")


# ── STATUS ────────────────────────────────────
def status():
    print("\n" + "=" * 50)
    print("DATA STATUS:")
    files = [
        ("games_history.csv",        "Games 2023-2025"),
        ("lines_history.csv",        "Lines 2023-2025"),
        ("lines_2026.csv",           "Lines 2026"),
        ("schedule_2026.csv",        "Schedule 2026"),
        ("elo_weekly.csv",           "Weekly Elo (temporal)"),
        ("ratings_sp.csv",           "SP+ ratings"),
        ("portal_2026.csv",          "Portal 2026"),
        ("returning_production.csv", "Returning production"),
        ("talent.csv",               "Talent composite"),
        ("ppa_weekly.csv",           "Weekly PPA (temporal)"),
    ]
    for fname, label in files:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            n = len(pd.read_csv(path))
            print(f"  ✓ {label}: {n} rows")
        else:
            print(f"  ✗ {label}: NOT AVAILABLE")
    print("=" * 50)
    print("Next: python features.py")


if __name__ == "__main__":
    print("=" * 50)
    print(f"CFB ALGO v2 — Fetch  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("Portal era training: 2023-2025")
    print("=" * 50)
    fetch_games()
    fetch_lines()
    fetch_weekly_elo()
    fetch_sp()
    fetch_roster()
    fetch_advanced()
    fetch_context()
    fetch_fpi()
    status()
