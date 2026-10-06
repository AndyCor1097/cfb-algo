"""
fetch_inseason.py — Pull ALL available 2026 in-season data from CFBD
Runs weekly after games finish to get the most complete picture possible.

Usage:
  python fetch_inseason.py
"""

import requests, pandas as pd, os, time, json
from datetime import datetime

CFBD_KEY = "b/c47r9EoNh7dOISJ+veiebAWqQIBphtZzdxz9UfdPG4b2GlaPmfKL9unFxI4+8R"
BASE     = "https://api.collegefootballdata.com"
HEADERS  = {"Authorization": f"Bearer {CFBD_KEY}", "Accept": "application/json"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
YEAR     = 2026


def fetch(endpoint, params=None, label=""):
    try:
        r = requests.get(f"{BASE}{endpoint}", headers=HEADERS, params=params or {}, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if data: print(f"  ✓ {label} — {len(data)} records")
            else:    print(f"  ~ {label} — empty")
            return data
        else:
            print(f"  ✗ [{r.status_code}] {label}")
            return []
    except Exception as e:
        print(f"  ✗ {label}: {e}"); return []
    finally:
        time.sleep(0.25)


def save(data, filename, label=""):
    if not data: return
    try:
        df = pd.json_normalize(data)
        df.to_csv(os.path.join(DATA_DIR, filename), index=False)
        print(f"    -> {filename} ({len(df)} rows)")
    except Exception as e:
        print(f"    ! Could not save {filename}: {e}")


def run():
    print("=" * 55)
    print(f"TheCFBAlgo — Full In-Season Fetch  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("Pulling everything CFBD has for 2026")
    print("=" * 55)

    # ── 1. SP+ Ratings (updated weekly) ──────────────
    print("\n[1] SP+ Ratings 2026...")
    sp = fetch("/ratings/sp", {"year": YEAR}, "SP+ 2026")
    if sp:
        df = pd.json_normalize(sp)
        df["year"] = YEAR
        df.to_csv(os.path.join(DATA_DIR, "ratings_sp_2026_current.csv"), index=False)
        print(f"    -> ratings_sp_2026_current.csv ({len(df)} teams)")

    # ── 2. Elo Ratings (weekly snapshots) ─────────────
    print("\n[2] Elo Ratings 2026 (all weeks)...")
    elo_rows = []
    for w in range(1, 17):
        data = fetch("/ratings/elo", {"year": YEAR, "week": w}, f"  Elo wk{w}")
        if not data: break
        elo_rows += [{**r, "year": YEAR, "week": w} for r in data]
    save(elo_rows, "elo_weekly_2026.csv")

    # ── 3. SRS Ratings ────────────────────────────────
    print("\n[3] SRS Ratings 2026...")
    srs = fetch("/ratings/srs", {"year": YEAR}, "SRS 2026")
    save(srs, "ratings_srs_2026.csv")

    # ── 4. FPI Ratings ────────────────────────────────
    print("\n[4] FPI Ratings 2026...")
    fpi = fetch("/ratings/fpi", {"year": YEAR}, "FPI 2026")
    save(fpi, "ratings_fpi_2026.csv")

    # ── 5. Season PPA (cumulative) ────────────────────
    print("\n[5] Season PPA 2026 (cumulative)...")
    ppa = fetch("/ppa/teams", {"year": YEAR}, "PPA 2026")
    save(ppa, "ppa_2026_current.csv")

    # ── 6. Weekly PPA ─────────────────────────────────
    print("\n[6] Weekly PPA 2026...")
    wppa = []
    for w in range(1, 17):
        data = fetch("/ppa/teams", {"year": YEAR, "week": w}, f"  PPA wk{w}")
        if not data: break
        wppa += [{**r, "year": YEAR, "week": w} for r in data]
    save(wppa, "ppa_weekly_2026.csv")

    # ── 7. Advanced Season Stats ───────────────────────
    print("\n[7] Advanced Stats 2026 (season)...")
    adv = fetch("/stats/season/advanced", {"year": YEAR}, "Advanced stats 2026")
    save(adv, "advanced_stats_2026.csv")

    # ── 8. Team Season Stats (standard) ───────────────
    print("\n[8] Team Season Stats 2026...")
    stats = fetch("/stats/season", {"year": YEAR}, "Season stats 2026")
    save(stats, "team_stats_2026.csv")

    # ── 9. Game Box Scores (actual performance/game) ──
    print("\n[9] Game Box Scores 2026 (per-game stats)...")
    box_rows = []
    for w in range(1, 17):
        data = fetch("/games/teams", {"year": YEAR, "week": w, "seasonType": "regular"},
                     f"  Box scores wk{w}")
        if not data: break
        box_rows += [{**r, "week": w} for r in data]
    save(box_rows, "game_box_scores_2026.csv")

    # ── 10. Team Records ──────────────────────────────
    print("\n[10] Team Records 2026...")
    records = fetch("/records", {"year": YEAR}, "Records 2026")
    save(records, "team_records_2026.csv")

    # ── 11. Rankings (AP Poll etc.) ───────────────────
    print("\n[11] Rankings 2026...")
    rankings_rows = []
    for w in range(1, 17):
        data = fetch("/rankings", {"year": YEAR, "week": w, "seasonType": "regular"},
                     f"  Rankings wk{w}")
        if not data: break
        rankings_rows += [{**r, "week": w} for r in data]
    save(rankings_rows, "rankings_2026.csv")

    # ── 12. Pregame Win Probability ───────────────────
    print("\n[12] Pregame Win Probability 2026...")
    wp_rows = []
    for w in range(1, 17):
        data = fetch("/metrics/wp/pregame", {"year": YEAR, "week": w},
                     f"  Win prob wk{w}")
        if not data: break
        wp_rows += [{**r, "week": w} for r in data]
    save(wp_rows, "pregame_wp_2026.csv")

    # ── 13. Drive Stats ───────────────────────────────
    print("\n[13] Drive Stats 2026...")
    drives = fetch("/drives", {"year": YEAR, "seasonType": "regular"}, "Drives 2026")
    save(drives, "drives_2026.csv")

    # ── 14. Player Season Stats (key positions) ────────
    print("\n[14] Player Stats 2026 (QB passing)...")
    qb = fetch("/stats/player/season",
               {"year": YEAR, "category": "passing"}, "QB passing 2026")
    save(qb, "player_stats_passing_2026.csv")

    print("\n[14b] Player Stats 2026 (rushing)...")
    rush = fetch("/stats/player/season",
                 {"year": YEAR, "category": "rushing"}, "Rushing 2026")
    save(rush, "player_stats_rushing_2026.csv")

    # ── 15. Talent Composite ──────────────────────────
    print("\n[15] Talent Composite 2026...")
    talent = fetch("/talent", {"year": YEAR}, "Talent 2026")
    save(talent, "talent_2026.csv")

    # ── 16. Returning Production ──────────────────────
    print("\n[16] Returning Production 2026...")
    ret = fetch("/player/returning", {"year": YEAR}, "Returning production 2026")
    save(ret, "returning_production_2026.csv")

    # ── Summary ───────────────────────────────────────
    print("\n" + "=" * 55)
    print("Fetch complete. Files saved to data/")
    print("Now run: python predict.py {week} --push")
    print("=" * 55)


if __name__ == "__main__":
    run()
