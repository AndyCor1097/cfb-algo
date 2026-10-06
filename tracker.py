"""
tracker.py — Log weekly picks and grade results
  python tracker.py           show summary
  python tracker.py log       log from cfb_predictions.xlsx
  python tracker.py grade     grade completed games
"""

import pandas as pd, numpy as np, os, sys
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV      = os.path.join(DATA_DIR, "cfb_tracker.csv")
UNIT     = 100
COLS     = ["date","season","week","home_team","away_team",
            "pred_home","pred_away","pred_margin","pred_total",
            "book_spread","model_edge","book_total","total_edge",
            "home_win_pct","book_home_ml","book_away_ml",
            "actual_home","actual_away",
            "spread_result","total_result","ml_result",
            "spread_pnl","total_pnl","ml_pnl","logged_at"]


def load():
    return pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame(columns=COLS)


def log():
    pred_path = os.path.join(os.path.dirname(__file__), "cfb_predictions.xlsx")
    if not os.path.exists(pred_path):
        print("  ! cfb_predictions.xlsx not found"); return
    preds = pd.read_excel(pred_path)
    tracker = load()
    new = []
    for _, r in preds.iterrows():
        new.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "season": 2026, "week": r.get("Wk",""),
            "home_team": r.get("Home",""), "away_team": r.get("Away",""),
            "pred_home": r.get("Pred Home",""), "pred_away": r.get("Pred Away",""),
            "pred_margin": r.get("Pred Margin",""), "pred_total": r.get("Pred Total",""),
            "book_spread": r.get("Book Spread",""), "model_edge": r.get("Model Edge",""),
            "book_total": r.get("Book Total",""), "total_edge": r.get("Total Edge",""),
            "home_win_pct": r.get("Home Win%",""),
            "book_home_ml": r.get("Book Home ML",""), "book_away_ml": r.get("Book Away ML",""),
            "actual_home": "", "actual_away": "",
            "spread_result": "PENDING", "total_result": "PENDING", "ml_result": "PENDING",
            "spread_pnl": "", "total_pnl": "", "ml_pnl": "",
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    tracker = pd.concat([tracker, pd.DataFrame(new)], ignore_index=True)
    tracker.drop_duplicates(subset=["home_team","away_team","week","season"], keep="last", inplace=True)
    tracker.to_csv(CSV, index=False)
    print(f"  -> Logged {len(new)} games to cfb_tracker.csv")


def grade():
    tracker = load()
    graded = 0
    for idx, row in tracker[tracker["spread_result"] == "PENDING"].iterrows():
        if not row.get("actual_home") or pd.isna(row.get("actual_home")): continue
        h, a = float(row["actual_home"]), float(row["actual_away"])
        margin = h - a
        total  = h + a

        # Spread
        spread = row.get("book_spread")
        pm     = row.get("pred_margin", 0)
        if not pd.isna(spread):
            model_home = float(pm) + float(spread) > 0
            actual_home = margin + float(spread) > 0
            win = model_home == actual_home
            tracker.at[idx, "spread_result"] = "W" if win else "L"
            tracker.at[idx, "spread_pnl"]    = UNIT * (100/110) if win else -UNIT

        # Total
        ou = row.get("book_total")
        pt = row.get("pred_total", 0)
        if not pd.isna(ou):
            model_over  = float(pt) > float(ou)
            actual_over = total > float(ou)
            win = model_over == actual_over
            tracker.at[idx, "total_result"] = "W" if win else "L"
            tracker.at[idx, "total_pnl"]    = UNIT * (100/110) if win else -UNIT

        # ML
        hwp   = row.get("home_win_pct", 50)
        hml   = row.get("book_home_ml")
        aml   = row.get("book_away_ml")
        hw    = margin > 0
        if not pd.isna(hml):
            book_prob = (-float(hml)/(-float(hml)+100)) if float(hml) < 0 else (100/(float(hml)+100))
            edge = float(hwp)/100 - book_prob
            if abs(edge) >= 0.05:
                take_home = edge > 0
                win = take_home == hw
                juice = float(hml) if take_home else float(aml)
                payout = UNIT*(100/abs(juice)) if (win and juice < 0) else (UNIT*(juice/100) if win else -UNIT)
                tracker.at[idx, "ml_result"] = "W" if win else "L"
                tracker.at[idx, "ml_pnl"]    = payout

        graded += 1

    tracker.to_csv(CSV, index=False)
    print(f"  Graded {graded} games")
    summary(tracker)


def summary(tracker=None):
    if tracker is None: tracker = load()
    print("\n" + "=" * 50)
    print("CFB TRACKER — 2026 Season")
    print("=" * 50)
    for market, col, pnl_col in [
        ("SPREAD", "spread_result", "spread_pnl"),
        ("TOTAL",  "total_result",  "total_pnl"),
        ("ML",     "ml_result",     "ml_pnl"),
    ]:
        done = tracker[tracker.get(col, pd.Series()) == "W"].append(
               tracker[tracker.get(col, pd.Series()) == "L"]) if col in tracker.columns else pd.DataFrame()
        done = tracker[tracker[col].isin(["W","L"])] if col in tracker.columns else pd.DataFrame()
        if done.empty:
            print(f"  {market}: No graded picks yet"); continue
        w = (done[col] == "W").sum()
        pnl = done[pnl_col].sum() if pnl_col in done.columns else 0
        print(f"  {market}: {w}-{len(done)-w} ({w/len(done):.1%}) | PnL: ${pnl:.0f}")

    print(f"\n  Total: {len(tracker)} | Pending: {(tracker.get('spread_result','') == 'PENDING').sum() if 'spread_result' in tracker.columns else 0}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "log": log()
    elif cmd == "grade": grade()
    else: summary()
