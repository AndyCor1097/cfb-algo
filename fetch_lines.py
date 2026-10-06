"""
fetch_lines.py — Pull CFB lines from Odds API, normalize team names to match CFBD
"""

import requests, pandas as pd, os
from datetime import datetime, timezone

ODDS_KEY = "2b08d1a670815553d9b505ee1f3e9549"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TEAM_NAME_MAP = {
    "Alabama Crimson Tide": "Alabama",
    "Akron Zips": "Akron",
    "Air Force Falcons": "Air Force",
    "Appalachian State Mountaineers": "App State",
    "Arizona State Sun Devils": "Arizona State",
    "Arizona Wildcats": "Arizona",
    "Arkansas Razorbacks": "Arkansas",
    "Arkansas State Red Wolves": "Arkansas State",
    "Army Black Knights": "Army",
    "Auburn Tigers": "Auburn",
    "Ball State Cardinals": "Ball State",
    "Baylor Bears": "Baylor",
    "Boise State Broncos": "Boise State",
    "Boston College Eagles": "Boston College",
    "Bowling Green Falcons": "Bowling Green",
    "BYU Cougars": "BYU",
    "Buffalo Bulls": "Buffalo",
    "California Golden Bears": "California",
    "Central Michigan Chippewas": "Central Michigan",
    "Charlotte 49ers": "Charlotte",
    "Cincinnati Bearcats": "Cincinnati",
    "Clemson Tigers": "Clemson",
    "Coastal Carolina Chanticleers": "Coastal Carolina",
    "Colorado Buffaloes": "Colorado",
    "Colorado State Rams": "Colorado State",
    "Connecticut Huskies": "UConn",
    "Duke Blue Devils": "Duke",
    "East Carolina Pirates": "East Carolina",
    "Eastern Michigan Eagles": "Eastern Michigan",
    "Florida Atlantic Owls": "Florida Atlantic",
    "Florida Gators": "Florida",
    "Florida International Panthers": "Florida International",
    "Florida State Seminoles": "Florida State",
    "Fresno State Bulldogs": "Fresno State",
    "Georgia Bulldogs": "Georgia",
    "Georgia Southern Eagles": "Georgia Southern",
    "Georgia State Panthers": "Georgia State",
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Hawaii Rainbow Warriors": "Hawai'i",
    "Howard Bison": "Howard",
    "Houston Cougars": "Houston",
    "Illinois Fighting Illini": "Illinois",
    "Indiana Hoosiers": "Indiana",
    "Iowa Hawkeyes": "Iowa",
    "Iowa State Cyclones": "Iowa State",
    "Jacksonville State Gamecocks": "Jacksonville State",
    "James Madison Dukes": "James Madison",
    "Kansas Jayhawks": "Kansas",
    "Kansas State Wildcats": "Kansas State",
    "Kennesaw State Owls": "Kennesaw State",
    "Kent State Golden Flashes": "Kent State",
    "Kentucky Wildcats": "Kentucky",
    "Liberty Flames": "Liberty",
    "Louisiana Ragin' Cajuns": "Louisiana",
    "Louisiana Ragin Cajuns": "Louisiana",
    "Louisiana Monroe Warhawks": "UL Monroe",
    "Louisiana Tech Bulldogs": "Louisiana Tech",
    "Louisville Cardinals": "Louisville",
    "LSU Tigers": "LSU",
    "Marshall Thundering Herd": "Marshall",
    "Maryland Terrapins": "Maryland",
    "Massachusetts Minutemen": "Massachusetts",
    "Memphis Tigers": "Memphis",
    "Miami Hurricanes": "Miami",
    "Miami FL": "Miami",
    "Miami (OH) RedHawks": "Miami (OH)",
    "Miami OH": "Miami (OH)",
    "Michigan Wolverines": "Michigan",
    "Michigan State Spartans": "Michigan State",
    "Middle Tennessee Blue Raiders": "Middle Tennessee",
    "Minnesota Golden Gophers": "Minnesota",
    "Mississippi State Bulldogs": "Mississippi State",
    "Missouri Tigers": "Missouri",
    "Missouri State Bears": "Missouri State",
    "Navy Midshipmen": "Navy",
    "NC State Wolfpack": "NC State",
    "Nebraska Cornhuskers": "Nebraska",
    "Nevada Wolf Pack": "Nevada",
    "New Mexico Lobos": "New Mexico",
    "New Mexico State Aggies": "New Mexico State",
    "North Carolina Tar Heels": "North Carolina",
    "North Texas Mean Green": "North Texas",
    "Northern Illinois Huskies": "Northern Illinois",
    "Northwestern Wildcats": "Northwestern",
    "Notre Dame Fighting Irish": "Notre Dame",
    "Ohio Bobcats": "Ohio",
    "Ohio State Buckeyes": "Ohio State",
    "Oklahoma Sooners": "Oklahoma",
    "Oklahoma State Cowboys": "Oklahoma State",
    "Old Dominion Monarchs": "Old Dominion",
    "Ole Miss Rebels": "Ole Miss",
    "Oregon Ducks": "Oregon",
    "Oregon State Beavers": "Oregon State",
    "Penn State Nittany Lions": "Penn State",
    "Pittsburgh Panthers": "Pittsburgh",
    "Purdue Boilermakers": "Purdue",
    "Rice Owls": "Rice",
    "Robert Morris Colonials": "Robert Morris",
    "Rutgers Scarlet Knights": "Rutgers",
    "Sacramento State Hornets": "Sacramento State",
    "Sam Houston Bearkats": "Sam Houston",
    "San Diego State Aztecs": "San Diego State",
    "San Jose State Spartans": "San José State",
    "San José State Spartans": "San José State",
    "SMU Mustangs": "SMU",
    "South Alabama Jaguars": "South Alabama",
    "South Carolina Gamecocks": "South Carolina",
    "South Florida Bulls": "South Florida",
    "Southern Miss Golden Eagles": "Southern Miss",
    "Stanford Cardinal": "Stanford",
    "Syracuse Orange": "Syracuse",
    "TCU Horned Frogs": "TCU",
    "Temple Owls": "Temple",
    "Tennessee Volunteers": "Tennessee",
    "Texas Longhorns": "Texas",
    "Texas A&M Aggies": "Texas A&M",
    "Texas State Bobcats": "Texas State",
    "Texas Tech Red Raiders": "Texas Tech",
    "Toledo Rockets": "Toledo",
    "Troy Trojans": "Troy",
    "Tulane Green Wave": "Tulane",
    "Tulsa Golden Hurricane": "Tulsa",
    "UAB Blazers": "UAB",
    "UCF Knights": "UCF",
    "UCLA Bruins": "UCLA",
    "ULM": "UL Monroe",
    "UMass": "Massachusetts",
    "UNLV Rebels": "UNLV",
    "USC Trojans": "USC",
    "UTEP Miners": "UTEP",
    "UTSA Roadrunners": "UTSA",
    "Utah Utes": "Utah",
    "Utah State Aggies": "Utah State",
    "Vanderbilt Commodores": "Vanderbilt",
    "Virginia Cavaliers": "Virginia",
    "Virginia Tech Hokies": "Virginia Tech",
    "Wake Forest Demon Deacons": "Wake Forest",
    "Washington Huskies": "Washington",
    "Washington State Cougars": "Washington State",
    "West Virginia Mountaineers": "West Virginia",
    "Western Kentucky Hilltoppers": "Western Kentucky",
    "Western Michigan Broncos": "Western Michigan",
    "Wisconsin Badgers": "Wisconsin",
    "Wyoming Cowboys": "Wyoming",
    "William and Mary Tribe": "William & Mary",
    "North Carolina Central Eagles": "NC Central",
}

# CFBD week boundaries — Thursday before Saturday games
WEEK_STARTS = {
    1:  datetime(2026, 8, 28,  tzinfo=timezone.utc),
    2:  datetime(2026, 9, 4,   tzinfo=timezone.utc),
    3:  datetime(2026, 9, 11,  tzinfo=timezone.utc),
    4:  datetime(2026, 9, 18,  tzinfo=timezone.utc),
    5:  datetime(2026, 9, 25,  tzinfo=timezone.utc),
    6:  datetime(2026, 10, 2,  tzinfo=timezone.utc),
    7:  datetime(2026, 10, 9,  tzinfo=timezone.utc),
    8:  datetime(2026, 10, 16, tzinfo=timezone.utc),
    9:  datetime(2026, 10, 23, tzinfo=timezone.utc),
    10: datetime(2026, 10, 30, tzinfo=timezone.utc),
    11: datetime(2026, 11, 6,  tzinfo=timezone.utc),
    12: datetime(2026, 11, 13, tzinfo=timezone.utc),
    13: datetime(2026, 11, 20, tzinfo=timezone.utc),
    14: datetime(2026, 11, 27, tzinfo=timezone.utc),
    15: datetime(2026, 12, 4,  tzinfo=timezone.utc),
}


def normalize_team(name):
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    return name


def get_week(commence):
    try:
        dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        for wk in sorted(WEEK_STARTS.keys(), reverse=True):
            if dt >= WEEK_STARTS[wk]:
                return wk
        return 1
    except:
        return None


def fetch_odds():
    print("Fetching CFB lines from Odds API...")
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/",
            params={
                "apiKey": ODDS_KEY, "regions": "us",
                "markets": "spreads,totals,h2h",
                "bookmakers": "draftkings,fanduel",
                "oddsFormat": "american",
            }, timeout=15
        )
        if r.status_code != 200:
            print(f"  ! {r.status_code}: {r.text[:100]}"); return []
        data = r.json()
        print(f"  ✓ {len(data)} games")
        return data
    except Exception as e:
        print(f"  ! {e}"); return []


def parse_lines(data):
    rows = []
    for game in data:
        home     = normalize_team(game.get("home_team", ""))
        away     = normalize_team(game.get("away_team", ""))
        commence = game.get("commence_time", "")
        week     = get_week(commence)

        for bm in game.get("bookmakers", []):
            provider = bm.get("key", "").replace("_", " ").title()
            spread = total = home_ml = away_ml = None
            for mkt in bm.get("markets", []):
                k = mkt.get("key")
                for o in mkt.get("outcomes", []):
                    nm  = normalize_team(o.get("name", ""))
                    pt  = o.get("point")
                    pr  = o.get("price")
                    if k == "spreads" and nm == home: spread  = pt
                    if k == "totals"  and o.get("name") == "Over": total = pt
                    if k == "h2h"    and nm == home: home_ml = pr
                    if k == "h2h"    and nm == away: away_ml = pr

            rows.append({
                "game_id":        f"odds_{home[:4]}_{away[:4]}_{commence[:10]}".replace(" ", "_"),
                "home_team":      home,
                "away_team":      away,
                "week":           week,
                "provider":       provider,
                "spread":         spread,
                "over_under":     total,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "source":         "odds_api",
                "commence_time":  commence,
            })
    return rows


def run():
    print("=" * 50)
    print(f"Fetch Lines — Odds API  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 50)

    data = fetch_odds()
    if not data: return

    rows = parse_lines(data)
    df   = pd.DataFrame(rows)
    if df.empty:
        print("  ! No lines parsed"); return

    # Show week breakdown
    by_week = df[df["provider"].str.lower()=="draftkings"].groupby("week")["game_id"].nunique()
    print(f"  DraftKings games by week:")
    for wk, cnt in by_week.items():
        try: print(f"    Week {int(wk)}: {cnt} games")
        except: pass

    # Preview current week
    dk = df[df["provider"].str.lower()=="draftkings"].copy()
    if not dk.empty:
        current_week = dk["week"].min()
        wk_games = dk[dk["week"]==current_week].groupby(["home_team","away_team"]).first().reset_index()
        print(f"\n  Week {int(current_week)} games ({len(wk_games)}):")
        for _, r in wk_games.head(10).iterrows():
            sp = f"{r['spread']:+.1f}" if pd.notna(r.get("spread")) else "N/A"
            ou = f"{r['over_under']}"   if pd.notna(r.get("over_under")) else "N/A"
            print(f"    {r['home_team']} vs {r['away_team']}: {sp} | O/U {ou}")

    # Merge with existing
    lines_path = os.path.join(DATA_DIR, "lines_2026.csv")
    if os.path.exists(lines_path):
        existing = pd.read_csv(lines_path, low_memory=False)
        if "source" in existing.columns:
            existing = existing[existing["source"] != "odds_api"].copy()
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, df], ignore_index=True)
    combined.to_csv(lines_path, index=False)

    total_by_week = combined.groupby("week")["game_id"].nunique() if "week" in combined.columns else None
    print(f"\n  -> lines_2026.csv updated ({len(combined)} rows)")
    if total_by_week is not None:
        for wk, cnt in total_by_week.items():
            try: print(f"    Week {int(wk)}: {cnt} games")
            except: pass

    print(f"\n  Run: python predict.py {int(current_week) if not dk.empty else 'N'} --push")


if __name__ == "__main__":
    run()
