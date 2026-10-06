"""
fetch_venues.py — Pull venue/stadium data and calculate home field advantage per team
Run once, updates rarely.
"""

import requests, pandas as pd, os, json
from datetime import datetime

CFBD_KEY = "b/c47r9EoNh7dOISJ+veiebAWqQIBphtZzdxz9UfdPG4b2GlaPmfKL9unFxI4+8R"
BASE     = "https://api.collegefootballdata.com"
HEADERS  = {"Authorization": f"Bearer {CFBD_KEY}", "Accept": "application/json"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Known elite home field environments — manual adjustments on top of capacity
ELITE_HFA = {
    "Auburn":          2.0,   # Jordan-Hare night games
    "LSU":             2.0,   # Death Valley night games
    "Penn State":      1.5,   # White Out
    "Alabama":         1.0,
    "Georgia":         1.0,
    "Ohio State":      1.0,
    "Michigan":        1.0,
    "Tennessee":       1.5,   # Neyland loud
    "Texas":           1.0,
    "Clemson":         1.5,   # Death Valley
    "Wisconsin":       1.0,
    "Iowa":            1.0,
    "Notre Dame":      1.0,
    "Oklahoma":        0.5,
    "Texas A&M":       1.5,   # 12th Man
    "Florida":         1.0,   # Swamp
    "West Virginia":   1.5,   # Mountaineer Field
    "Kansas State":    1.0,
    "Boise State":     1.0,   # Blue turf factor
}

# Dome stadiums — weather neutral, slight reduction in HFA vs outdoor
DOME_TEAMS = ["Georgia Dome", "Mercedes-Benz Superdome"]


def run():
    print("=" * 50)
    print(f"Fetch Venues  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 50)

    # Pull venues
    r = requests.get(f"{BASE}/venues", headers=HEADERS, timeout=15)
    venues = r.json()
    print(f"  {len(venues)} venues")

    # Pull teams to get venue associations
    r2 = requests.get(f"{BASE}/teams", headers=HEADERS,
                      params={"division": "fbs"}, timeout=15)
    teams = r2.json()
    print(f"  {len(teams)} FBS teams")

    # Build venue lookup
    venue_map = {v["id"]: v for v in venues}

    # Build team -> venue -> capacity mapping
    rows = []
    for t in teams:
        name     = t.get("school","")
        venue_id = t.get("location", {}).get("venueId") if t.get("location") else None
        venue    = venue_map.get(venue_id, {}) if venue_id else {}
        capacity = venue.get("capacity", 0) or 0
        dome     = venue.get("dome", False) or False
        timezone = venue.get("timezone","") or ""

        # Base HFA from capacity (bigger stadium = more crowd noise = more advantage)
        # Scale: 10k=1.0, 50k=2.5, 100k=4.0 (roughly)
        if capacity >= 100000:
            base_hfa = 4.0
        elif capacity >= 80000:
            base_hfa = 3.5
        elif capacity >= 60000:
            base_hfa = 3.0
        elif capacity >= 40000:
            base_hfa = 2.5
        elif capacity >= 25000:
            base_hfa = 2.0
        elif capacity >= 15000:
            base_hfa = 1.5
        else:
            base_hfa = 1.0

        # Dome reduction — weather neutral, less noise impact
        if dome:
            base_hfa *= 0.85

        # Elite environment bonus
        elite_bonus = ELITE_HFA.get(name, 0)
        total_hfa   = round(base_hfa + elite_bonus, 2)

        rows.append({
            "team":      name,
            "venue":     venue.get("name",""),
            "capacity":  capacity,
            "dome":      dome,
            "timezone":  timezone,
            "base_hfa":  base_hfa,
            "elite_bonus": elite_bonus,
            "hfa":       total_hfa,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("hfa", ascending=False)

    print(f"\n  Top 15 home field advantages:")
    for _, r in df.head(15).iterrows():
        print(f"    {r['team']:<25} {r['hfa']:.1f} pts  ({int(r['capacity']):,} cap)")

    df.to_csv(os.path.join(DATA_DIR, "home_field_advantage.csv"), index=False)
    print(f"\n  -> home_field_advantage.csv saved ({len(df)} teams)")


if __name__ == "__main__":
    run()
