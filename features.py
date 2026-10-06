"""
features.py — Build feature matrix
2026 predictions use current in-season stats as primary signal
Historical training uses weekly PPA snapshots (prior week, no leakage)
"""

import pandas as pd, numpy as np, os

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_YEARS = [2023, 2024, 2025]
CURR_YEAR   = 2026

LEAKAGE_COLS = [
    "home_score","away_score","homePoints","awayPoints",
    "homeLineScores","awayLineScores",
    "homePostgameWinProbability","awayPostgameWinProbability",
    "homePregameElo","awayPregameElo","homePostgameElo","awayPostgameElo",
    "excitementIndex","highlights","notes",
    "actual_margin","actual_total","cover_margin","went_over","home_win",
]


def load(f):
    p = os.path.join(DATA_DIR, f)
    return pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()


# ── TEAM RATINGS ──────────────────────────────
def get_sp(year):
    if year == CURR_YEAR:
        sp = load("ratings_sp_2026_current.csv")
        if not sp.empty:
            sp = sp.rename(columns={"rating":"sp","offense.rating":"sp_off","defense.rating":"sp_def"})
            return sp[[c for c in ["team","sp","sp_off","sp_def"] if c in sp.columns]]
    sp = load("ratings_sp.csv")
    if sp.empty: return pd.DataFrame()
    yr = sp[sp["year"]==year].copy()
    if yr.empty: yr = sp[sp["year"]==year-1].copy()
    yr = yr.rename(columns={"rating":"sp","offense.rating":"sp_off","defense.rating":"sp_def"})
    return yr[[c for c in ["team","sp","sp_off","sp_def"] if c in yr.columns]]


def get_elo(year, week):
    if year == CURR_YEAR:
        elo = load("elo_weekly_2026.csv")
        if not elo.empty and "week" in elo.columns:
            prior = elo[elo["week"] < week]
            if not prior.empty:
                snap = prior[prior["week"] == prior["week"].max()]
                return snap[["team","elo"]].copy()
    elo = load("elo_weekly.csv")
    if elo.empty: return pd.DataFrame()
    yr = elo[elo["year"]==year]
    prior = yr[yr["week"] < week]
    if prior.empty:
        # preseason — use prior year final
        prior_yr = elo[elo["year"]==year-1]
        if not prior_yr.empty:
            snap = prior_yr[prior_yr["week"]==prior_yr["week"].max()]
            return snap[["team","elo"]].copy()
        return pd.DataFrame()
    snap = prior[prior["week"]==prior["week"].max()]
    return snap[["team","elo"]].copy()


def get_ppa(year, week):
    """Get PPA using prior week data — no leakage."""
    if year == CURR_YEAR:
        ppa = load("ppa_weekly_2026.csv")
        if not ppa.empty and "week" in ppa.columns:
            prior = ppa[ppa["week"] < week]
            if not prior.empty:
                snap = prior[prior["week"]==prior["week"].max()]
                rename = {
                    "offense.overall":"ppa_off","defense.overall":"ppa_def",
                    "offense.passing":"ppa_off_pass","offense.rushing":"ppa_off_rush",
                    "defense.passing":"ppa_def_pass","defense.rushing":"ppa_def_rush",
                }
                snap = snap.rename(columns={k:v for k,v in rename.items() if k in snap.columns})
                keep = ["team"] + [v for v in rename.values() if v in snap.columns]
                return snap[[c for c in keep if c in snap.columns]]
        # Fall back to season cumulative
        ppa = load("ppa_2026_current.csv")
        if not ppa.empty:
            rename = {
                "offense.overall":"ppa_off","defense.overall":"ppa_def",
                "offense.passing":"ppa_off_pass","offense.rushing":"ppa_off_rush",
                "defense.passing":"ppa_def_pass","defense.rushing":"ppa_def_rush",
            }
            ppa = ppa.rename(columns={k:v for k,v in rename.items() if k in ppa.columns})
            keep = ["team"] + [v for v in rename.values() if v in ppa.columns]
            return ppa[[c for c in keep if c in ppa.columns]]
        return pd.DataFrame()
    else:
        ppa = load("ppa_weekly.csv")
        if ppa.empty: return pd.DataFrame()
        yr = ppa[ppa["year"]==year]
        prior = yr[yr["week"] < week]
        if prior.empty: return pd.DataFrame()
        snap = prior[prior["week"]==prior["week"].max()]
        rename = {
            "offense.overall":"ppa_off","defense.overall":"ppa_def",
            "offense.passing":"ppa_off_pass","offense.rushing":"ppa_off_rush",
            "defense.passing":"ppa_def_pass","defense.rushing":"ppa_def_rush",
        }
        snap = snap.rename(columns={k:v for k,v in rename.items() if k in snap.columns})
        keep = ["team"] + [v for v in rename.values() if v in snap.columns]
        return snap[[c for c in keep if c in snap.columns]]


def get_advanced(year):
    """Season-to-date advanced stats."""
    if year == CURR_YEAR:
        adv = load("advanced_stats_2026.csv")
    else:
        adv = load("advanced_stats.csv")
        if not adv.empty and "year" in adv.columns:
            adv = adv[adv["year"]==year].copy()
    if adv.empty: return pd.DataFrame()
    rename = {
        "offense.ppa":"adv_off_ppa","defense.ppa":"adv_def_ppa",
        "offense.successRate":"adv_off_sr","defense.successRate":"adv_def_sr",
        "offense.explosiveness":"adv_off_exp","defense.explosiveness":"adv_def_exp",
        "defense.havoc.total":"adv_havoc",
    }
    adv = adv.rename(columns={k:v for k,v in rename.items() if k in adv.columns})
    keep = ["team"] + [v for v in rename.values() if v in adv.columns]
    return adv[[c for c in keep if c in adv.columns]]


def get_roster(year):
    ret     = load("returning_production.csv")
    portal  = load("portal_2026.csv") if year==CURR_YEAR else pd.DataFrame()
    talent  = load("talent.csv")
    coaches = load("coaches.csv")
    features = {}

    if not ret.empty:
        ret_yr = ret[ret["year"]==year]
        if ret_yr.empty and year==CURR_YEAR:
            ret_yr = ret[ret["year"]==year-1]
        pct_map = {
            "percentPPA.offense":"ret_off","percentPPA.defense":"ret_def",
            "percentPPA.pass":"ret_pass","percentPPA.rush":"ret_rush",
        }
        for _, row in ret_yr.iterrows():
            team = row.get("team")
            if pd.isna(team): continue
            features[team] = {v: row.get(k,np.nan) for k,v in pct_map.items()}

    if not portal.empty and "destination" in portal.columns:
        arr = portal.groupby("destination").size().rename("portal_in")
        dep = portal.groupby("origin").size().rename("portal_out")
        net = pd.concat([arr,dep],axis=1).fillna(0)
        net["portal_net"] = net["portal_in"] - net["portal_out"]
        for team, row in net.iterrows():
            if team not in features: features[team] = {}
            features[team]["portal_net"] = row["portal_net"]

    if not talent.empty:
        t_yr = talent[talent["year"]==year]
        if t_yr.empty: t_yr = talent[talent["year"]==year-1]
        scol = "school" if "school" in t_yr.columns else "team"
        if scol in t_yr.columns and "talent" in t_yr.columns:
            for _, row in t_yr.iterrows():
                team = row[scol]
                if team not in features: features[team] = {}
                features[team]["talent"] = row["talent"]

    if not coaches.empty and "team" in coaches.columns:
        curr = coaches[coaches["year"]==year][["team","firstName","lastName"]]
        prev = coaches[coaches["year"]==year-1][["team","firstName","lastName"]]
        if not curr.empty and not prev.empty:
            merged = curr.merge(prev,on=["team","firstName","lastName"],how="left",indicator=True)
            merged["same_hc"] = (merged["_merge"]=="both").astype(int)
            for _, row in merged.iterrows():
                team = row["team"]
                if team not in features: features[team] = {}
                features[team]["same_hc"] = row["same_hc"]

    return pd.DataFrame(features).T.reset_index().rename(columns={"index":"team"})


def get_best_lines(lines_df, game_ids):
    if lines_df.empty: return pd.DataFrame()
    lines_df = lines_df[lines_df["game_id"].isin(game_ids)].copy()
    if lines_df.empty: return pd.DataFrame()
    lines_df["_p"] = lines_df["provider"].str.lower().str.strip()
    coverage = lines_df.groupby("_p")["game_id"].nunique()
    max_cov  = coverage.max()
    for p in ["consensus","fanduel","draftkings","espn bet",
              "william hill (new jersey)","bovada","caesars","teamrankings"]:
        if p in coverage.index and coverage[p] >= max_cov * 0.5:
            sub = lines_df[lines_df["_p"]==p]
            return sub.drop(columns=["_p"]).groupby("game_id").first().reset_index()
    best = coverage.idxmax()
    return lines_df[lines_df["_p"]==best].drop(columns=["_p"]).groupby("game_id").first().reset_index()


def merge_team_stats(df, stats, side, suffix):
    """Merge team stats for home or away side."""
    if stats.empty: return df
    cols = [c for c in stats.columns if c != "team"]
    renamed = stats.rename(columns={"team": side, **{c: f"{c}_{suffix}" for c in cols}})
    return df.merge(renamed, on=side, how="left")


def build_features_for_year(year):
    print(f"\n  Building features for {year}...")

    games = load("games_history.csv") if year < CURR_YEAR else load("schedule_2026.csv")
    if games.empty:
        print(f"    ! No games"); return pd.DataFrame()

    if year < CURR_YEAR and "season" in games.columns:
        games = games[games["season"]==year].copy()
        if "seasonType" in games.columns:
            games = games[games["seasonType"]=="regular"].copy()

    games = games.rename(columns={
        "homeTeam":"home_team","awayTeam":"away_team",
        "homePoints":"home_score","awayPoints":"away_score",
        "home_points":"home_score","away_points":"away_score",
        "neutralSite":"neutral_site","id":"game_id",
    })
    if "game_id" in games.columns:
        games = games.drop_duplicates(subset=["game_id"])

    # FBS only
    if "homeClassification" in games.columns:
        before = len(games)
        games  = games[(games["homeClassification"]=="fbs")&(games["awayClassification"]=="fbs")].copy()
        print(f"    FBS: {before} -> {len(games)} games")

    scores = games[[c for c in ["game_id","home_score","away_score"] if c in games.columns]].copy()

    # Preseason / season-level features (same all year)
    sp     = get_sp(year)
    roster = get_roster(year)
    adv    = get_advanced(year)

    # Build week by week for temporal Elo + PPA
    all_weeks = []
    weeks = sorted(games["week"].dropna().unique()) if "week" in games.columns else [1]

    for week in weeks:
        wk  = games[games["week"]==week].copy() if "week" in games.columns else games.copy()
        elo = get_elo(year, int(week))
        ppa = get_ppa(year, int(week))

        wk = merge_team_stats(wk, sp,     "home_team", "h")
        wk = merge_team_stats(wk, sp,     "away_team", "a")
        wk = merge_team_stats(wk, elo,    "home_team", "h")
        wk = merge_team_stats(wk, elo,    "away_team", "a")
        wk = merge_team_stats(wk, roster, "home_team", "h")
        wk = merge_team_stats(wk, roster, "away_team", "a")
        wk = merge_team_stats(wk, ppa,    "home_team", "h")
        wk = merge_team_stats(wk, ppa,    "away_team", "a")
        wk = merge_team_stats(wk, adv,    "home_team", "h")
        wk = merge_team_stats(wk, adv,    "away_team", "a")

        all_weeks.append(wk)

    df = pd.concat(all_weeks, ignore_index=True) if all_weeks else pd.DataFrame()
    if df.empty: return pd.DataFrame()

    # Differential features
    stat_bases = ["sp","sp_off","sp_def","elo","talent","same_hc",
                  "ret_off","ret_def","portal_net",
                  "ppa_off","ppa_def","ppa_off_pass","ppa_off_rush",
                  "adv_off_ppa","adv_def_ppa","adv_off_sr","adv_def_sr",
                  "adv_off_exp","adv_def_exp","adv_havoc"]
    for stat in stat_bases:
        h, a = f"{stat}_h", f"{stat}_a"
        if h in df.columns and a in df.columns:
            df[f"{stat}_diff"] = pd.to_numeric(df[h],errors="coerce") - pd.to_numeric(df[a],errors="coerce")

    # Game context
    if "neutral_site" in df.columns:
        df["neutral_site"] = pd.to_numeric(df["neutral_site"],errors="coerce").fillna(0).astype(int)
    else:
        df["neutral_site"] = 0
    df["home_field"] = (df["neutral_site"]==0).astype(int)
    df["season"]     = year

    # ATS history (prior year)
    ats = load("ats_history.csv")
    if not ats.empty and "team" in ats.columns:
        ats_yr = ats[ats["year"]==year-1].copy()
        if {"ats_wins","ats_losses"}.issubset(ats_yr.columns):
            ats_yr["ats_pct"] = ats_yr["ats_wins"]/(ats_yr["ats_wins"]+ats_yr["ats_losses"]+0.01)
            h_ats = ats_yr[["team","ats_pct"]].rename(columns={"team":"home_team","ats_pct":"ats_pct_h"})
            a_ats = ats_yr[["team","ats_pct"]].rename(columns={"team":"away_team","ats_pct":"ats_pct_a"})
            df = df.merge(h_ats,on="home_team",how="left").merge(a_ats,on="away_team",how="left")
            df["ats_pct_diff"] = df.get("ats_pct_h",0) - df.get("ats_pct_a",0)

    # Drop leakage
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors="ignore")

    # Lines
    lines_file = "lines_history.csv" if year < CURR_YEAR else "lines_2026.csv"
    lines = load(lines_file)
    if not lines.empty and "game_id" in df.columns:
        if year < CURR_YEAR and "year" in lines.columns:
            lines = lines[lines["year"]==year]
        best = get_best_lines(lines, df["game_id"].tolist())
        if not best.empty:
            for col in ["spread","over_under","provider","home_moneyline","away_moneyline"]:
                if col in df.columns: df.drop(columns=[col],inplace=True)
            df = df.merge(best[[c for c in ["game_id","spread","over_under","provider",
                                             "home_moneyline","away_moneyline"] if c in best.columns]],
                          on="game_id",how="left")
            df["spread"]     = pd.to_numeric(df.get("spread"),errors="coerce")
            df["over_under"] = pd.to_numeric(df.get("over_under"),errors="coerce")

    # Targets
    if year < CURR_YEAR and not scores.empty and "game_id" in df.columns:
        df = df.merge(scores,on="game_id",how="left")
        df["home_score"]    = pd.to_numeric(df.get("home_score"),errors="coerce")
        df["away_score"]    = pd.to_numeric(df.get("away_score"),errors="coerce")
        df["actual_margin"] = df["home_score"] - df["away_score"]
        df["actual_total"]  = df["home_score"] + df["away_score"]
        df["cover_margin"]  = df["actual_margin"] + df["spread"]
        df["went_over"]     = (df["actual_total"] > df["over_under"]).astype(float)
        df["home_win"]      = (df["actual_margin"] > 0).astype(float)

    out_path = os.path.join(DATA_DIR, f"features_{year}.csv")
    df.to_csv(out_path, index=False)
    n_spread = len(df.dropna(subset=["cover_margin"])) if "cover_margin" in df.columns else 0
    print(f"    -> features_{year}.csv ({len(df)} games, {n_spread} with spread)")
    return df


def build_all():
    print("="*50)
    print("CFB ALGO — Feature Engineering")
    print("In-season 2026 stats as primary signal")
    print("="*50)

    frames = []
    for y in TRAIN_YEARS:
        df = build_features_for_year(y)
        if not df.empty: frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        path = os.path.join(DATA_DIR, "features_all.csv")
        combined.to_csv(path, index=False)
        n_spread = len(combined.dropna(subset=["cover_margin"])) if "cover_margin" in combined.columns else 0
        print(f"\n  -> features_all.csv ({len(combined)} games, {n_spread} with spread)")

    build_features_for_year(CURR_YEAR)
    print("\nNext: python model.py")


if __name__ == "__main__":
    build_all()
