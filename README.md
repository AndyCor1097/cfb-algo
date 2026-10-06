# CFB Algo v2 — 2026

## What's new vs v1
- Temporally correct: weekly Elo snapshots, preseason SP+
- Dual score models: predicts home score + away score independently
- All three markets: spread, total, moneyline
- Walk-forward backtest: real week-by-week validation 2023-2025
- FBS filter, portal era training only (2023-2025)

## Setup
pip install requests pandas scikit-learn joblib openpyxl

## First time
python fetch_cfb.py    (takes ~10 mins — pulls weekly Elo for all years)
python features.py
python model.py
python backtest.py     (validate before betting)
python predict.py 1    (Week 1 picks)

## Weekly workflow
Monday: daily_run.bat → enter week number
Saturday: fill scores in cfb_tracker.csv → python tracker.py grade

## Tracker
python tracker.py          show record
python tracker.py log      log current predictions
python tracker.py grade    grade completed games (fill actual_home/away first)

## Run order when retraining
retrain.bat   (features + model + backtest)
