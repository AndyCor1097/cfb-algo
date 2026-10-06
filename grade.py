"""
grade.py — Auto-grade predictions by pulling results from CFBD
Run after games finish each week.

Usage:
  python grade.py 1    Grade Week 1
  python grade.py      Grade all weeks
"""

import requests, json, os, sys
from datetime import datetime

CFBD_KEY     = "b/c47r9EoNh7dOISJ+veiebAWqQIBphtZzdxz9UfdPG4b2GlaPmfKL9unFxI4+8R"
BASE         = "https://api.collegefootballdata.com"
HEADERS      = {"Authorization": f"Bearer {CFBD_KEY}", "Accept": "application/json"}
MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))
TRACKER_FILE = os.path.join(MODEL_DIR, "tracker.json")
WEEK_FILTER  = int(sys.argv[1]) if len(sys.argv) > 1 else None


def fetch_results(week, year=2026):
    try:
        r = requests.get(f"{BASE}/games", headers=HEADERS,
                         params={"year": year, "week": week, "seasonType": "regular"}, timeout=15)
        if r.status_code == 200:
            return [g for g in r.json() if g.get("homePoints") is not None]
        print(f"  ! CFBD {r.status_code}")
        return []
    except Exception as e:
        print(f"  ! {e}"); return []


def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE) as f: return json.load(f)
    return {"games": [], "last_updated": None}


def save_tracker(data):
    data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(TRACKER_FILE, "w") as f: json.dump(data, f, indent=2)


def load_predictions(week):
    path = os.path.join(MODEL_DIR, f"predictions_week{week}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f: return json.load(f).get("games", [])


def norm(name):
    fixes = {
        "Ole Miss": "Mississippi", "USC": "Southern California",
        "LSU": "Louisiana State", "TCU": "Texas Christian",
        "UCF": "Central Florida", "UAB": "Alabama Birmingham",
        "UNLV": "Nevada Las Vegas", "SMU": "Southern Methodist",
        "BYU": "Brigham Young", "Pitt": "Pittsburgh",
        "Miami (OH)": "Miami Ohio", "UL Monroe": "Louisiana Monroe",
        "NC State": "North Carolina State", "UConn": "Connecticut",
        "USF": "South Florida", "FIU": "Florida International",
        "FAU": "Florida Atlantic", "WKU": "Western Kentucky",
        "NIU": "Northern Illinois", "UMass": "Massachusetts",
        "App State": "Appalachian State", "N Dakota St": "North Dakota State",
        "Jax State": "Jacksonville State", "E Michigan": "Eastern Michigan",
        "Sacramento St": "Sacramento State",
    }
    for k, v in fixes.items(): name = name.replace(k, v)
    return name.lower().strip()


def grade_week(week):
    print(f"\n  Grading Week {week}...")
    predictions = load_predictions(week)
    if not predictions: return []

    results = fetch_results(week)
    if not results:
        print(f"  ! No completed games for Week {week} yet"); return []

    print(f"  {len(predictions)} predictions | {len(results)} completed games")

    # Build lookup
    lookup = {}
    for g in results:
        h = norm(g.get("homeTeam",""))
        a = norm(g.get("awayTeam",""))
        lookup[f"{h}|{a}"] = g
        lookup[f"{a}|{h}"] = g

    graded = []
    for pred in predictions:
        home = pred.get("home_team","")
        away = pred.get("away_team","")
        key  = f"{norm(home)}|{norm(away)}"

        result = lookup.get(key)
        if not result:
            # Partial match
            for k, v in lookup.items():
                parts = k.split("|")
                if norm(home)[:5] in parts[0] and norm(away)[:5] in parts[1]:
                    result = v; break

        if not result:
            pred["graded"] = False
            pred["spread_result"] = "PENDING"
            pred["total_result"]  = "PENDING"
            pred["ml_result"]     = "PENDING"
            graded.append(pred)
            continue

        # Scores
        ah = result.get("homePoints")
        aa = result.get("awayPoints")
        if norm(result.get("homeTeam","")) != norm(home):
            ah, aa = aa, ah

        actual_margin = ah - aa
        actual_total  = ah + aa

        pred["actual_home"]   = ah
        pred["actual_away"]   = aa
        pred["actual_margin"] = actual_margin
        pred["actual_total"]  = actual_total
        pred["graded"]        = True

        # Spread
        sp = pred.get("book_spread")
        if sp is not None:
            se = pred.get("spread_edge", 0) or 0
            model_home = se > 0
            home_covered = actual_margin + sp > 0
            if actual_margin + sp == 0:
                pred["spread_result"] = "PUSH"
            else:
                pred["spread_result"] = "W" if (model_home == home_covered) else "L"
            pred["spread_pick"] = "home" if model_home else "away"
        else:
            pred["spread_result"] = "NO LINE"

        # Total
        ou = pred.get("book_total")
        if ou is not None:
            model_over = (pred.get("pred_total") or 0) > ou
            went_over  = actual_total > ou
            if actual_total == ou:
                pred["total_result"] = "PUSH"
            else:
                pred["total_result"] = "W" if (model_over == went_over) else "L"
            pred["total_pick"] = "over" if model_over else "under"
        else:
            pred["total_result"] = "NO LINE"

        # ML
        hml = pred.get("home_moneyline")
        aml = pred.get("away_moneyline")
        hwp = pred.get("home_win_prob", 50) or 50
        if hml and aml:
            book_prob = (-float(hml)/(-float(hml)+100)) if float(hml)<0 else (100/(float(hml)+100))
            edge = hwp/100 - book_prob
            if abs(edge) >= 0.05:
                take_home = edge > 0
                pred["ml_result"] = "W" if (take_home == (actual_margin > 0)) else "L"
                pred["ml_pick"]   = "home" if take_home else "away"
            else:
                pred["ml_result"] = "NO EDGE"
        else:
            pred["ml_result"] = "NO LINE"

        graded.append(pred)

    done = [g for g in graded if g.get("graded")]
    print(f"  Graded: {len(done)} | Pending: {len(graded)-len(done)}")
    return graded


def record(games, col):
    w = len([g for g in games if g.get(col) == "W"])
    l = len([g for g in games if g.get(col) == "L"])
    p = len([g for g in games if g.get(col) == "PUSH"])
    pct = round(w/(w+l)*100, 1) if w+l > 0 else 0
    return {"W": w, "L": l, "P": p, "pct": pct}


def build_summary(games):
    graded = [g for g in games if g.get("graded")]
    weeks  = sorted(set(g.get("week") for g in graded if g.get("week")))
    by_week = []
    for wk in weeks:
        wk_g = [g for g in graded if g.get("week") == wk]
        by_week.append({
            "week": wk, "games": len(wk_g),
            "spread": record(wk_g, "spread_result"),
            "total":  record(wk_g, "total_result"),
            "ml":     record(wk_g, "ml_result"),
        })
    recent = sorted(
        [g for g in graded if g.get("spread_result") in ["W","L"]],
        key=lambda x: x.get("week", 0), reverse=True
    )[:20]
    return {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "overall": {
            "spread": record(graded, "spread_result"),
            "total":  record(graded, "total_result"),
            "ml":     record(graded, "ml_result"),
        },
        "by_week":        by_week,
        "recent_results": recent,
    }


def print_summary(games):
    graded = [g for g in games if g.get("graded")]
    print("\n" + "=" * 50)
    print("TRACKER SUMMARY")
    print("=" * 50)
    for market, col in [("SPREAD","spread_result"),("TOTAL","total_result"),("ML","ml_result")]:
        r = record(graded, col)
        if r["W"] + r["L"] == 0:
            print(f"  {market}: No graded picks yet")
        else:
            push_str = f"-{r['P']}" if r["P"] else ""
            print(f"  {market}: {r['W']}-{r['L']}{push_str} ({r['pct']}%)")
    weeks = sorted(set(g.get("week") for g in graded if g.get("week")))
    if len(weeks) > 1:
        print("\n  By Week (Spread):")
        for wk in weeks:
            wk_g = [g for g in graded if g.get("week") == wk]
            r = record(wk_g, "spread_result")
            if r["W"] + r["L"] > 0:
                print(f"    Week {wk}: {r['W']}-{r['L']} ({r['pct']}%)")


def run():
    print("=" * 50)
    print(f"TheCFBAlgo — Auto Grader  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 50)

    tracker = load_tracker()
    existing = {f"{g['home_team']}_{g['away_team']}_{g['week']}": g
                for g in tracker["games"]}

    weeks = [WEEK_FILTER] if WEEK_FILTER else [w for w in range(0,16)
             if os.path.exists(os.path.join(MODEL_DIR, f"predictions_week{w}.json"))]

    for week in weeks:
        graded = grade_week(week)
        for g in graded:
            key = f"{g['home_team']}_{g['away_team']}_{g['week']}"
            existing[key] = g

    tracker["games"] = list(existing.values())
    save_tracker(tracker)

    summary = build_summary(list(existing.values()))
    with open(os.path.join(MODEL_DIR, "tracker_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(list(existing.values()))
    print(f"\n  -> tracker.json updated")
    print(f"  -> tracker_summary.json updated")
    print(f"\n  Push to update site:")
    print(f"  git add tracker_summary.json && git commit -m 'Update tracker' && git push")


if __name__ == "__main__":
    run()
