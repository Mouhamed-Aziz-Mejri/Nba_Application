"""
train_similarity.py
-------------------
Trains a KNN-based player similarity model using cosine distance
on per-game stats from database_24_25.csv.

Run from the project root:
    python train_similarity.py

Output:
    predictor/model/similarity_model.pkl   ← KNN model + scaler + player index
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "predictor", "data", "database_24_25.csv")
INFO_PATH  = os.path.join(BASE_DIR, "predictor", "data", "player_info.json")
MODEL_PATH = os.path.join(BASE_DIR, "predictor", "model", "similarity_model.pkl")

# ── Features used for similarity ─────────────────────────────────────────────
# Chosen to capture playing style: scoring, playmaking, defense, efficiency
FEATURES = ["PTS", "AST", "TRB", "STL", "BLK", "TOV", "MP", "FG%", "3P%", "FT%"]

print("=" * 55)
print("  NBA Player Similarity Model — Training")
print("=" * 55)

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"\n📂 Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Normalise column names
col_map  = {c.lower(): c for c in df.columns}
STANDARD = {
    "Player": ["player"],
    "Tm":     ["tm", "team"],
    "MP":     ["mp", "min", "minutes"],
    "PTS":    ["pts", "points"],
    "TRB":    ["trb", "reb", "rebounds"],
    "AST":    ["ast", "assists"],
    "STL":    ["stl", "steals"],
    "BLK":    ["blk", "blocks"],
    "TOV":    ["tov", "to", "turnovers"],
    "FG%":    ["fg%", "fgpct", "fg_pct"],
    "3P%":    ["3p%", "3ppct", "3p_pct"],
    "FT%":    ["ft%", "ftpct", "ft_pct"],
}
for standard, variants in STANDARD.items():
    if standard not in df.columns:
        for v in variants:
            if v in col_map:
                df = df.rename(columns={col_map[v]: standard})
                break

# Remove TOT rows (traded player league-wide aggregates)
if "Tm" in df.columns:
    df = df[df["Tm"] != "TOT"].copy()

print(f"   Raw rows: {len(df)}")

# ── Aggregate: one row per player (average across all teams/games) ─────────────
group_cols = ["Player"]
stat_cols  = [c for c in FEATURES if c in df.columns]
pct_cols   = [c for c in ["FG%", "3P%", "FT%"] if c in df.columns]

# Detect totals vs per-game
is_totals = df["PTS"].median() > 100 if "PTS" in df.columns else False
print(f"   Data type: {'Season totals' if is_totals else 'Per-game averages'}")

# Count real games played per player
games_played = df.groupby("Player").size().reset_index(name="G")

agg_dict = {}
for c in [c for c in stat_cols if c not in pct_cols]:
    agg_dict[c] = "sum" if is_totals else "mean"
for c in pct_cols:
    if c in df.columns:
        agg_dict[c] = "mean"

df_agg = df.groupby("Player", as_index=False).agg(agg_dict)
df_agg = df_agg.merge(games_played, on="Player", how="left")

# If totals → divide by games to get per-game
if is_totals:
    for c in [c for c in stat_cols if c not in pct_cols]:
        if c in df_agg.columns:
            df_agg[c] = (df_agg[c] / df_agg["G"].replace(0, np.nan)).round(2)

# Normalise pct cols (51.2 → 0.512 if needed)
for c in pct_cols:
    if c in df_agg.columns:
        df_agg[c] = df_agg[c].apply(
            lambda v: round(v / 100, 3) if pd.notna(v) and v > 1 else (round(v, 3) if pd.notna(v) else 0.0)
        )

# ── Load player info (Pos/Age) ────────────────────────────────────────────────
player_info = {}
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, encoding="utf-8") as f:
        player_info = json.load(f)
    print(f"   Player info cache: {len(player_info)} entries")

import re, unicodedata

def normalize_name(name):
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"[''`.]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip())
    return re.sub(r"\s+", " ", name).strip()

info_norm = {normalize_name(k): v for k, v in player_info.items()}

def lookup(name):
    return player_info.get(name) or info_norm.get(normalize_name(name)) or {}

df_agg["Pos"] = df_agg["Player"].apply(lambda n: lookup(n).get("Pos"))
df_agg["Age"] = df_agg["Player"].apply(lambda n: lookup(n).get("Age"))

print(f"   Players after aggregation: {len(df_agg)}")

# ── Filter: keep only players with enough data ────────────────────────────────
# Require at least 5 games played and non-null core stats
available_features = [c for c in FEATURES if c in df_agg.columns]
df_model = df_agg.dropna(subset=available_features[:4]).copy()  # need at least PTS/AST/REB/STL
df_model = df_model[df_model["G"] >= 5].copy()
df_model[available_features] = df_model[available_features].fillna(0.0)

print(f"   Players after filtering (≥5 games): {len(df_model)}")

# ── Scale features ────────────────────────────────────────────────────────────
X = df_model[available_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n📐 Features used: {available_features}")
print(f"   Feature matrix shape: {X_scaled.shape}")

# ── Train KNN model ───────────────────────────────────────────────────────────
# cosine metric captures style similarity better than euclidean
# n_neighbors=11 → return top 10 (excluding the player itself)
k = min(11, len(df_model))
knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
knn.fit(X_scaled)

print(f"\n🤖 KNN model trained (k={k}, metric=cosine)")

# ── Build player index ────────────────────────────────────────────────────────
# Store everything needed to look up a player and return results
players_list = []
for _, row in df_model.iterrows():
    players_list.append({
        "name":  row["Player"],
        "Pos":   row.get("Pos"),
        "Age":   int(float(row["Age"])) if pd.notna(row.get("Age", None)) and str(row.get("Age","")).replace(".","").isdigit() else None,
        "G":     int(row["G"]),
        "PTS":   round(float(row["PTS"]), 1) if "PTS" in row else None,
        "AST":   round(float(row["AST"]), 1) if "AST" in row else None,
        "TRB":   round(float(row["TRB"]), 1) if "TRB" in row else None,
        "STL":   round(float(row["STL"]), 1) if "STL" in row else None,
        "BLK":   round(float(row["BLK"]), 1) if "BLK" in row else None,
        "MP":    round(float(row["MP"]),  1) if "MP"  in row else None,
        "FG%":   round(float(row["FG%"]), 3) if "FG%" in row else None,
        "3P%":   round(float(row["3P%"]), 3) if "3P%" in row else None,
        "FT%":   round(float(row["FT%"]), 3) if "FT%" in row else None,
    })

# ── Save model bundle ─────────────────────────────────────────────────────────
bundle = {
    "knn":      knn,
    "scaler":   scaler,
    "features": available_features,
    "players":  players_list,           # list aligned with X_scaled rows
    "X_scaled": X_scaled,               # pre-computed for fast lookup
}

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(bundle, MODEL_PATH)

print(f"\n✅ Model saved to {MODEL_PATH}")
print(f"   Bundle contains: knn, scaler, {len(available_features)} features, {len(players_list)} players")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\n🧪 Sanity check — similar players to 'Stephen Curry':")
curry_idx = next((i for i, p in enumerate(players_list) if "Curry" in p["name"] and "Stephen" in p["name"]), None)
if curry_idx is not None:
    distances, indices = knn.kneighbors([X_scaled[curry_idx]])
    for rank, (dist, idx) in enumerate(zip(distances[0][1:6], indices[0][1:6]), 1):
        p = players_list[idx]
        sim = round((1 - dist) * 100, 1)
        print(f"   {rank}. {p['name']:25s}  {p.get('Pos','?'):2s}  {sim}% similar")
else:
    print("   Stephen Curry not found in dataset (check player name)")

print("\n🏀 Training complete!")