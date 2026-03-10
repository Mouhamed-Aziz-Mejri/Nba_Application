import joblib
import numpy as np
import os
import json
import re
import unicodedata
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.shortcuts import render

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(os.path.dirname(__file__), "model", "nba_model.pkl")
DATA_PATH        = os.path.join(os.path.dirname(__file__), "data",  "database_24_25.csv")
PLAYER_INFO_PATH = os.path.join(os.path.dirname(__file__), "data",  "player_info.json")

# ── Load model ───────────────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ NBA model loaded!")
else:
    print(f"⚠️  Model not found at {MODEL_PATH}.")

# ── Load player info cache (Age + Pos from internet) ─────────────────────────
player_info      = {}
player_info_norm = {}   # normalized-name → info, for fuzzy lookup

def _normalize_name(name):
    """Normalize player name: remove accents, suffixes (Jr/Sr/II/III), punctuation."""
    if not name:
        return ""
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"[''`\.]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip())
    name = re.sub(r"\s+", " ", name).strip()
    return name

if os.path.exists(PLAYER_INFO_PATH):
    with open(PLAYER_INFO_PATH, "r", encoding="utf-8") as f:
        player_info = json.load(f)
    player_info_norm = {_normalize_name(k): v for k, v in player_info.items()}
    print(f"✅ Player info cache loaded: {len(player_info)} players ({len(player_info_norm)} normalized)")
else:
    print(f"⚠️  No player_info.json found. Run: python manage.py fetch_player_info")

def _lookup_player(name):
    """Look up player info: exact match first, then normalized fuzzy match."""
    if name in player_info:
        return player_info[name]
    norm = _normalize_name(name)
    return player_info_norm.get(norm, {})

# ── Load & preprocess dataset ─────────────────────────────────────────────────
df     = None
df_avg = None

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # ── Normalise column names ────────────────────────────────────────────
    col_map  = {c.lower(): c for c in df.columns}
    STANDARD = {
        "Player": ["player"],
        "Tm":     ["tm", "team"],
        "Pos":    ["pos", "position"],
        "Age":    ["age"],
        "MP":     ["mp", "min", "minutes", "mpg"],
        "PTS":    ["pts", "points", "ppg"],
        "TRB":    ["trb", "reb", "rebounds", "rpg", "total_reb"],
        "AST":    ["ast", "assists", "apg"],
        "STL":    ["stl", "steals", "spg"],
        "BLK":    ["blk", "blocks", "bpg"],
        "TOV":    ["tov", "to", "turnovers"],
        "FG%":    ["fg%", "fgpct", "fg_pct", "field_goal_pct"],
        "3P%":    ["3p%", "3ppct", "3p_pct", "three_point_pct"],
        "FT%":    ["ft%", "ftpct", "ft_pct", "free_throw_pct"],
    }
    rename = {}
    for standard, variants in STANDARD.items():
        if standard not in df.columns:
            for v in variants:
                if v in col_map:
                    rename[col_map[v]] = standard
                    break
    if rename:
        df = df.rename(columns=rename)

    # Remove TOT rows
    if "Tm" in df.columns:
        df = df[df["Tm"] != "TOT"].copy()

    stat_cols_avg = [c for c in ["MP", "PTS", "TRB", "AST", "STL", "BLK", "TOV"] if c in df.columns]
    pct_cols      = [c for c in ["FG%", "3P%", "FT%"] if c in df.columns]
    group_cols    = [c for c in ["Player", "Tm"] if c in df.columns]

    # Detect season totals vs per-game
    is_totals = False
    if "PTS" in df.columns:
        median_pts = df["PTS"].median()
        is_totals  = median_pts > 100
        print(f"📊 Median PTS={median_pts:.1f} → {'TOTALS' if is_totals else 'PER-GAME'}")

    # Count actual rows per player+team = real games played
    games_played = df.groupby(group_cols).size().reset_index(name="G")

    # Build averaged stats
    agg_dict = {}
    for c in ["Pos", "Age"]:
        if c in df.columns and df[c].notna().sum() > 0:
            agg_dict[c] = "first"

    if is_totals:
        for c in stat_cols_avg:
            agg_dict[c] = "sum"
    else:
        for c in stat_cols_avg:
            agg_dict[c] = "mean"
    for c in pct_cols:
        agg_dict[c] = "mean"

    df_avg = df.groupby(group_cols, as_index=False).agg(agg_dict)
    df_avg = df_avg.merge(games_played, on=group_cols, how="left")

    if is_totals:
        for c in stat_cols_avg:
            if c in df_avg.columns:
                df_avg[c] = (df_avg[c] / df_avg["G"].replace(0, np.nan)).round(1)
    else:
        for c in stat_cols_avg:
            if c in df_avg.columns:
                df_avg[c] = df_avg[c].round(1)

    # ── Inject Age + Pos from player_info cache ───────────────────────────
    if player_info:
        df_avg["Pos"] = df_avg["Player"].apply(lambda n: _lookup_player(n).get("Pos") or None)
        df_avg["Age"] = df_avg["Player"].apply(lambda n: _lookup_player(n).get("Age") or None)
        filled = df_avg["Pos"].notna().sum()
        print(f"✅ Injected Pos/Age for {filled}/{len(df_avg)} players from cache")

    # ── Clean types ───────────────────────────────────────────────────────
    if "Age" in df_avg.columns:
        df_avg["Age"] = pd.to_numeric(df_avg["Age"], errors="coerce")
        df_avg["Age"] = df_avg["Age"].apply(lambda x: int(x) if pd.notna(x) and x != 0 else None)

    df_avg["G"] = df_avg["G"].fillna(0).astype(int)

    # FIX: handle NaN floats before calling .split() on Pos
    if "Pos" in df_avg.columns:
        def clean_pos(x):
            if pd.isna(x) or x is None:
                return None
            s = str(x).strip()
            if s in ("None", "nan", ""):
                return None
            return s.split("-")[0].strip()
        df_avg["Pos"] = df_avg["Pos"].apply(clean_pos)

    for c in pct_cols:
        if c in df_avg.columns:
            df_avg[c] = df_avg[c].apply(
                lambda v: round(v / 100, 3) if pd.notna(v) and v > 1 else (round(v, 3) if pd.notna(v) else None)
            )

    print(f"✅ df_avg: {df_avg.shape[0]} rows | cols: {df_avg.columns.tolist()}")
else:
    print(f"⚠️  Dataset not found at {DATA_PATH}.")


# ── Load similarity model ────────────────────────────────────────────────────
SIM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "similarity_model.pkl")
sim_bundle = None
if os.path.exists(SIM_MODEL_PATH):
    sim_bundle = joblib.load(SIM_MODEL_PATH)
    print(f"✅ Similarity model loaded: {len(sim_bundle['players'])} players")
else:
    print(f"⚠️  No similarity model found. Run: python train_similarity.py")

# ── Helper ────────────────────────────────────────────────────────────────────
def get_tier(score: float) -> str:
    if score >= 80:   return "Elite"
    elif score >= 65: return "All-Star"
    elif score >= 50: return "Starter"
    elif score >= 35: return "Rotation"
    else:             return "Bench"


# ── HTML pages ────────────────────────────────────────────────────────────────
def index(request):
    return render(request, "predictor/index.html")

def login_page(request):
    return render(request, "login.html")

def similar_page(request):
    q = request.GET.get("q", "")
    return render(request, "predictor/similar.html", {"initial_query": q})

def teams_page(request):
    return render(request, "predictor/teams.html")

def roster_page(request, team_code):
    return render(request, "predictor/roster.html", {"team_code": team_code})


# ── API ───────────────────────────────────────────────────────────────────────
@method_decorator(csrf_exempt, name='dispatch')
class HealthView(APIView):
    def get(self, request):
        sample = []
        if df_avg is not None:
            # Replace NaN/inf with None so JSON serialization works
            sample = df_avg.head(3).where(df_avg.head(3).notna(), other=None).to_dict(orient="records")
        return Response({
            "status":            "ok",
            "model_loaded":      model is not None,
            "dataset_loaded":    df_avg is not None,
            "player_info_cache": len(player_info),
            "columns":           df_avg.columns.tolist() if df_avg is not None else [],
            "sample":            sample,
        })


@method_decorator(csrf_exempt, name='dispatch')
class TeamsView(APIView):
    def get(self, request):
        if df_avg is None:
            return Response({"error": "Dataset not loaded."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        teams = (
            df_avg.groupby("Tm")["Player"]
            .count()
            .reset_index()
            .rename(columns={"Tm": "team", "Player": "player_count"})
            .sort_values("team")
            .to_dict(orient="records")
        )
        return Response({"teams": teams, "total": len(teams)})


@method_decorator(csrf_exempt, name='dispatch')
class RosterView(APIView):
    def get(self, request, team_code):
        if df_avg is None:
            return Response({"error": "Dataset not loaded."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        roster_df = df_avg[df_avg["Tm"] == team_code.upper()].copy()
        if roster_df.empty:
            return Response({"error": f"Team '{team_code}' not found."}, status=status.HTTP_404_NOT_FOUND)

        if "PTS" in roster_df.columns:
            roster_df = roster_df.sort_values("PTS", ascending=False)

        stat_cols = ["Pos", "Age", "G", "MP", "PTS", "TRB", "AST", "STL", "BLK", "TOV", "FG%", "3P%", "FT%"]
        available = [c for c in stat_cols if c in roster_df.columns]

        players = []
        for _, row in roster_df.iterrows():
            player = {"name": row["Player"]}
            for col in available:
                val = row[col]
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    player[col] = None
                elif col in ("Age", "G"):
                    player[col] = int(float(val)) if val else None
                else:
                    player[col] = val
            players.append(player)

        return Response({
            "team":         team_code.upper(),
            "player_count": len(players),
            "players":      players,
        })


@method_decorator(csrf_exempt, name='dispatch')
class SimilarPlayersView(APIView):
    """GET /api/similar/<player_name>/?n=5 — Return N most similar players."""
    def get(self, request, player_name):
        if sim_bundle is None:
            return Response(
                {"error": "Similarity model not loaded. Run: python train_similarity.py"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        players  = sim_bundle["players"]
        X_scaled = sim_bundle["X_scaled"]
        knn      = sim_bundle["knn"]
        n        = min(int(request.GET.get("n", 5)), 10)

        # Find player index — exact → case-insensitive → normalized fuzzy → partial
        q_norm = _normalize_name(player_name)

        idx = next((i for i, p in enumerate(players) if p["name"] == player_name), None)
        if idx is None:
            idx = next((i for i, p in enumerate(players) if p["name"].lower() == player_name.lower()), None)
        if idx is None:
            idx = next((i for i, p in enumerate(players) if _normalize_name(p["name"]) == q_norm), None)
        if idx is None:
            idx = next((i for i, p in enumerate(players) if player_name.lower() in p["name"].lower()), None)
        if idx is None:
            # Return closest suggestions
            suggestions = sorted(
                [p["name"] for p in players if q_norm[:4] in _normalize_name(p["name"])],
            )[:8]
            return Response({
                "error":       f"Player '{player_name}' not found in similarity model.",
                "suggestions": suggestions or sorted([p["name"] for p in players])[:20],
                "hint":        "Use /api/players/?q=name to search."
            }, status=status.HTTP_404_NOT_FOUND)

        # KNN query
        distances, indices = knn.kneighbors([X_scaled[idx]])

        def clean(v):
            """Replace NaN/inf floats with None for JSON safety."""
            if v is None:
                return None
            try:
                if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                    return None
            except Exception:
                pass
            return v

        query_player = players[idx]
        similar = []
        for dist, i in zip(distances[0][1:n+1], indices[0][1:n+1]):
            p   = players[i]
            sim = round((1 - float(dist)) * 100, 1)
            similar.append({
                "name":       p["name"],
                "similarity": sim,
                "Pos":        clean(p.get("Pos")),
                "Age":        clean(p.get("Age")),
                "PTS":        clean(p.get("PTS")),
                "AST":        clean(p.get("AST")),
                "TRB":        clean(p.get("TRB")),
                "STL":        clean(p.get("STL")),
                "BLK":        clean(p.get("BLK")),
                "MP":         clean(p.get("MP")),
                "FG%":        clean(p.get("FG%")),
                "3P%":        clean(p.get("3P%")),
                "FT%":        clean(p.get("FT%")),
            })

        q = query_player
        return Response({
            "query":   {
                "name": q["name"],
                "Pos":  clean(q.get("Pos")),
                "PTS":  clean(q.get("PTS")),
                "AST":  clean(q.get("AST")),
            },
            "similar": similar,
            "total":   len(similar),
        })


@method_decorator(csrf_exempt, name='dispatch')
class PlayersSearchView(APIView):
    """GET /api/players/?q=curry — Search player names for autocomplete."""
    def get(self, request):
        if sim_bundle is None:
            return Response({"error": "Similarity model not loaded."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        q       = request.GET.get("q", "").lower().strip()
        players = sim_bundle["players"]
        if q:
            # fuzzy: check normalized name too so "doncic" finds "Luka Dončić"
            matches = [
                p["name"] for p in players
                if q in p["name"].lower() or q in _normalize_name(p["name"])
            ]
        else:
            matches = [p["name"] for p in players]
        matches.sort()
        return Response({"players": matches, "total": len(matches)})


@method_decorator(csrf_exempt, name='dispatch')
class PredictView(APIView):
    def post(self, request):
        if model is None:
            return Response({"error": "Model not loaded."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        data = request.data
        required_fields = ["AST", "STL", "BLK", "FG%", "3P%", "FT%", "MP", "TOV"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return Response({"error": f"Missing fields: {missing}"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            features = np.array([[
                float(data["AST"]), float(data["STL"]), float(data["BLK"]),
                float(data["FG%"]), float(data["3P%"]), float(data["FT%"]),
                float(data["MP"]),  float(data["TOV"]),
            ]])
            score       = float(model.predict(features)[0])
            tier        = get_tier(score)
            player_name = data.get("player_name", "Player")
            return Response({
                "player_name":       player_name,
                "performance_score": round(score, 2),
                "performance_tier":  tier,
                "message": f"{player_name} is projected as a {tier}-level performer with a score of {round(score, 2)}."
            })
        except ValueError as e:
            return Response({"error": f"Invalid value: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)