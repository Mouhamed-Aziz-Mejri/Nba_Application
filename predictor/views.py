import joblib
import numpy as np
import os
import json
import re
import unicodedata
import pandas as pd

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication

# ── Override DRF SessionAuthentication to skip CSRF enforcement ───────────────
class CsrfExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return   # skip CSRF check completely

# Apply globally to all APIView subclasses
APIView.authentication_classes = [CsrfExemptSessionAuthentication]

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(os.path.dirname(__file__), "model", "nba_model.pkl")
DATA_PATH        = os.path.join(os.path.dirname(__file__), "data",  "database_24_25.csv")
PLAYER_INFO_PATH = os.path.join(os.path.dirname(__file__), "data",  "player_info.json")

# ── Load model ────────────────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ NBA model loaded!")
else:
    print(f"⚠️  Model not found at {MODEL_PATH}.")

# ── Player info cache ─────────────────────────────────────────────────────────
player_info      = {}
player_info_norm = {}

def _normalize_name(name):
    """Normalize player name for cache/similarity lookups ONLY. Never touches prediction values."""
    if not name:
        return ""
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"[''`.]", "", name)
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

    if "Tm" in df.columns:
        df = df[df["Tm"] != "TOT"].copy()

    stat_cols_avg = [c for c in ["MP", "PTS", "TRB", "AST", "STL", "BLK", "TOV"] if c in df.columns]
    pct_cols      = [c for c in ["FG%", "3P%", "FT%"] if c in df.columns]
    group_cols    = [c for c in ["Player", "Tm"] if c in df.columns]

    is_totals = False
    if "PTS" in df.columns:
        median_pts = df["PTS"].median()
        is_totals  = median_pts > 100
        print(f"📊 Median PTS={median_pts:.1f} → {'TOTALS' if is_totals else 'PER-GAME'}")

    games_played = df.groupby(group_cols).size().reset_index(name="G")

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

    if player_info:
        df_avg["Pos"] = df_avg["Player"].apply(lambda n: _lookup_player(n).get("Pos") or None)
        df_avg["Age"] = df_avg["Player"].apply(lambda n: _lookup_player(n).get("Age") or None)
        filled = df_avg["Pos"].notna().sum()
        print(f"✅ Injected Pos/Age for {filled}/{len(df_avg)} players from cache")

    if "Age" in df_avg.columns:
        df_avg["Age"] = pd.to_numeric(df_avg["Age"], errors="coerce")
        df_avg["Age"] = df_avg["Age"].apply(lambda x: int(x) if pd.notna(x) and x != 0 else None)

    df_avg["G"] = df_avg["G"].fillna(0).astype(int)

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

# ── Load similarity model ─────────────────────────────────────────────────────
SIM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "similarity_model.pkl")
sim_bundle = None
if os.path.exists(SIM_MODEL_PATH):
    sim_bundle = joblib.load(SIM_MODEL_PATH)
    print(f"✅ Similarity model loaded: {len(sim_bundle['players'])} players")
else:
    print(f"⚠️  No similarity model found. Run: python train_similarity.py")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_tier(score: float) -> str:
    if score >= 80:   return "Elite"
    elif score >= 65: return "All-Star"
    elif score >= 50: return "Starter"
    elif score >= 35: return "Rotation"
    else:             return "Bench"


# ══════════════════════════════════════════════════════════════════════════════
# AUTH VIEWS
# ══════════════════════════════════════════════════════════════════════════════

def login_page(request):
    """Show login page. Redirect to home if already authenticated."""
    if request.user.is_authenticated:
        return redirect("index")
    return render(request, "login.html")


@require_POST
@csrf_exempt
def login_view(request):
    """POST /auth/login/ — Authenticate and create session."""
    try:
        body     = json.loads(request.body)
        email    = body.get("email", "").strip().lower()
        password = body.get("password", "")
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"error": "Invalid request body."}, status=400)

    if not email or not password:
        return JsonResponse({"error": "Email and password are required."}, status=400)

    # Look up user by email (Django uses username internally)
    try:
        user_obj = User.objects.get(email=email)
        username = user_obj.username
    except User.DoesNotExist:
        return JsonResponse({"error": "No account found with this email."}, status=401)

    user = authenticate(request, username=username, password=password)
    if user is None:
        return JsonResponse({"error": "Incorrect password."}, status=401)

    if not user.is_active:
        return JsonResponse({"error": "This account has been disabled."}, status=403)

    login(request, user)
    return JsonResponse({
        "ok":       True,
        "username": user.username,
        "email":    user.email,
        "redirect": "/",
    })


@require_POST
@csrf_exempt
def register_view(request):
    """POST /auth/register/ — Create account and auto-login."""
    try:
        body     = json.loads(request.body)
        name     = body.get("name", "").strip()
        email    = body.get("email", "").strip().lower()
        password = body.get("password", "")
        confirm  = body.get("confirm", "")
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"error": "Invalid request body."}, status=400)

    # Validation
    if not name or not email or not password:
        return JsonResponse({"error": "All fields are required."}, status=400)

    if len(password) < 8:
        return JsonResponse({"error": "Password must be at least 8 characters."}, status=400)

    if password != confirm:
        return JsonResponse({"error": "Passwords do not match."}, status=400)

    if User.objects.filter(email=email).exists():
        return JsonResponse({"error": "An account with this email already exists."}, status=409)

    # Build a clean username from name
    base_username = re.sub(r"[^a-z0-9]", "", name.lower().replace(" ", "_"))[:20] or "user"
    username      = base_username
    suffix        = 1
    while User.objects.filter(username=username).exists():
        username = f"{base_username}{suffix}"
        suffix  += 1

    # Create user
    user = User.objects.create_user(
        username   = username,
        email      = email,
        password   = password,
        first_name = name.split()[0] if name else "",
        last_name  = " ".join(name.split()[1:]) if len(name.split()) > 1 else "",
    )

    login(request, user)
    return JsonResponse({
        "ok":       True,
        "username": user.username,
        "email":    user.email,
        "redirect": "/",
    }, status=201)


@require_POST
@csrf_exempt
def logout_view(request):
    """POST /auth/logout/ — Destroy session."""
    logout(request)
    return JsonResponse({"ok": True, "redirect": "/login/"})


@csrf_exempt
def me_view(request):
    """GET /auth/me/ — Return current user info (used by frontend)."""
    if request.user.is_authenticated:
        return JsonResponse({
            "authenticated": True,
            "username":      request.user.username,
            "email":         request.user.email,
            "full_name":     request.user.get_full_name(),
        })
    return JsonResponse({"authenticated": False}, status=401)


# ══════════════════════════════════════════════════════════════════════════════
# HTML PAGE VIEWS  (all protected by @login_required)
# ══════════════════════════════════════════════════════════════════════════════

@login_required(login_url="/login/")
def index(request):
    return render(request, "predictor/index.html")

@login_required(login_url="/login/")
def similar_page(request):
    q = request.GET.get("q", "")
    return render(request, "predictor/similar.html", {"initial_query": q})

@login_required(login_url="/login/")
def teams_page(request):
    return render(request, "predictor/teams.html")

@login_required(login_url="/login/")
def roster_page(request, team_code):
    return render(request, "predictor/roster.html", {"team_code": team_code})


# ══════════════════════════════════════════════════════════════════════════════
# API VIEWS
# ══════════════════════════════════════════════════════════════════════════════

@method_decorator(csrf_exempt, name="dispatch")
class HealthView(APIView):
    def get(self, request):
        sample = []
        if df_avg is not None:
            sample = df_avg.head(3).where(df_avg.head(3).notna(), other=None).to_dict(orient="records")
        return Response({
            "status":            "ok",
            "model_loaded":      model is not None,
            "dataset_loaded":    df_avg is not None,
            "player_info_cache": len(player_info),
            "columns":           df_avg.columns.tolist() if df_avg is not None else [],
            "sample":            sample,
        })


@method_decorator(csrf_exempt, name="dispatch")
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


@method_decorator(csrf_exempt, name="dispatch")
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
        return Response({"team": team_code.upper(), "player_count": len(players), "players": players})


@method_decorator(csrf_exempt, name="dispatch")
class SimilarPlayersView(APIView):
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
        q_norm   = _normalize_name(player_name)
        idx = next((i for i, p in enumerate(players) if p["name"] == player_name), None)
        if idx is None:
            idx = next((i for i, p in enumerate(players) if p["name"].lower() == player_name.lower()), None)
        if idx is None:
            idx = next((i for i, p in enumerate(players) if _normalize_name(p["name"]) == q_norm), None)
        if idx is None:
            idx = next((i for i, p in enumerate(players) if player_name.lower() in p["name"].lower()), None)
        if idx is None:
            suggestions = sorted([p["name"] for p in players if q_norm[:4] in _normalize_name(p["name"])])[:8]
            return Response({
                "error":       f"Player '{player_name}' not found.",
                "suggestions": suggestions or sorted([p["name"] for p in players])[:20],
            }, status=status.HTTP_404_NOT_FOUND)

        distances, indices = knn.kneighbors([X_scaled[idx]])

        def clean(v):
            if v is None: return None
            try:
                if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                    return None
            except Exception:
                pass
            return v

        q = players[idx]
        similar = []
        for dist, i in zip(distances[0][1:n+1], indices[0][1:n+1]):
            p = players[i]
            similar.append({
                "name":       p["name"],
                "similarity": round((1 - float(dist)) * 100, 1),
                "Pos": clean(p.get("Pos")), "Age": clean(p.get("Age")),
                "PTS": clean(p.get("PTS")), "AST": clean(p.get("AST")),
                "TRB": clean(p.get("TRB")), "STL": clean(p.get("STL")),
                "BLK": clean(p.get("BLK")), "MP":  clean(p.get("MP")),
                "FG%": clean(p.get("FG%")), "3P%": clean(p.get("3P%")),
                "FT%": clean(p.get("FT%")),
            })
        return Response({
            "query":   {"name": q["name"], "Pos": clean(q.get("Pos")), "PTS": clean(q.get("PTS")), "AST": clean(q.get("AST"))},
            "similar": similar,
            "total":   len(similar),
        })


@method_decorator(csrf_exempt, name="dispatch")
class PlayersSearchView(APIView):
    def get(self, request):
        if sim_bundle is None:
            return Response({"error": "Similarity model not loaded."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        q       = request.GET.get("q", "").lower().strip()
        players = sim_bundle["players"]
        if q:
            matches = [p["name"] for p in players if q in p["name"].lower() or q in _normalize_name(p["name"])]
        else:
            matches = [p["name"] for p in players]
        matches.sort()
        return Response({"players": matches, "total": len(matches)})


@method_decorator(csrf_exempt, name="dispatch")
class PredictView(APIView):
    def post(self, request):
        if model is None:
            return Response({"error": "Model not loaded."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        data     = request.data
        required = ["AST", "STL", "BLK", "FG%", "3P%", "FT%", "MP", "TOV"]
        missing  = [f for f in required if f not in data]
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
                "message": f"{player_name} is projected as a {tier}-level performer with a score of {round(score, 2)}.",
            })
        except ValueError as e:
            return Response({"error": f"Invalid value: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# ══════════════════════════════════════════════════════════════════════════════
# PLAYOFF VIEWS — append these to the bottom of views.py
# ══════════════════════════════════════════════════════════════════════════════

PLAYOFF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model",  "playoff_model.pkl")
TEAMS_PATH         = os.path.join(os.path.dirname(__file__), "data",   "teams_stats_24_25.json")

# ── Load playoff model + teams data at startup ────────────────────────────────
playoff_bundle = None
teams_data     = {}

if os.path.exists(PLAYOFF_MODEL_PATH):
    playoff_bundle = joblib.load(PLAYOFF_MODEL_PATH)
    print(f"✅ Playoff model loaded | CV acc: {playoff_bundle['cv_accuracy']:.1%}")
else:
    print("⚠️  No playoff model found. Run: python train_playoff_model.py")

if os.path.exists(TEAMS_PATH):
    with open(TEAMS_PATH, encoding="utf-8") as f:
        teams_data = json.load(f)
    print(f"✅ Teams data loaded: {len(teams_data.get('teams', {}))} teams")
else:
    print(f"⚠️  No teams_stats_24_25.json found at {TEAMS_PATH}")


def _build_playoff_features(home_code, away_code, teams):
    """Build feature vector for a series matchup."""
    h = teams[home_code]
    a = teams[away_code]
    net_diff  = h["net_rtg"]     - a["net_rtg"]
    seed_diff = a["seed"]        - h["seed"]
    return {
        "seed_diff":                 seed_diff,
        "net_rtg_diff":              net_diff,
        "off_rtg_diff":              h["off_rtg"]      - a["off_rtg"],
        "def_rtg_diff":              a["def_rtg"]      - h["def_rtg"],
        "W_diff":                    h["W"]            - a["W"],
        "pace_diff":                 h["pace"]         - a["pace"],
        "exp_diff":                  h["playoff_exp"]  - a["playoff_exp"],
        "rest_diff":                 0,
        "quality_seed_interaction":  net_diff * seed_diff,
        "off_vs_def":                h["off_rtg"]      - a["def_rtg"],
        "def_vs_off":                a["def_rtg"]      - h["off_rtg"],
    }


def _predict_series(home_code, away_code, teams, bundle):
    """Return (predicted_winner, prob_home_wins, series_probs_per_game)."""
    feats    = _build_playoff_features(home_code, away_code, teams)
    features = bundle["features"]
    X_row    = pd.DataFrame([feats])[features]
    prob_home = float(bundle["model"].predict_proba(X_row)[0][1])
    winner    = home_code if prob_home >= 0.5 else away_code
    win_prob  = prob_home if prob_home >= 0.5 else (1 - prob_home)

    # Simulate game-by-game probability for a best-of-7
    # P(winning series in N games) using binomial approach
    p = prob_home  # prob home wins any single game
    game_probs = _series_game_distribution(p)

    return winner, round(win_prob * 100, 1), round(prob_home * 100, 1), game_probs


def _series_game_distribution(p):
    """
    Estimate probability distribution over series length (4,5,6,7 games).
    Uses the negative binomial: P(team wins series in exactly n games).
    Returns dict with expected games and win probability per game count.
    """
    from math import comb
    results = {}
    # P(home wins in exactly n games) for n in 4..7
    home_by_game = {}
    away_by_game = {}
    for n in range(4, 8):
        # Home wins: need 4 wins, last game is a win, won 3 of first n-1
        home_by_game[n] = comb(n-1, 3) * (p**4) * ((1-p)**(n-4))
        away_by_game[n] = comb(n-1, 3) * ((1-p)**4) * (p**(n-4))

    total_home = sum(home_by_game.values())
    total_away = sum(away_by_game.values())

    expected = sum(n * (home_by_game[n] + away_by_game[n]) for n in range(4, 8))
    results["expected_games"] = round(expected, 1)
    results["by_games"] = {
        str(n): round((home_by_game[n] + away_by_game[n]) * 100, 1)
        for n in range(4, 8)
    }
    return results


def _simulate_full_bracket(teams):
    """Simulate the entire 2024-25 playoff bracket, return structured results."""
    bundle  = playoff_bundle
    bracket = teams_data.get("bracket", {})
    results = {"East": {}, "West": {}, "Finals": {}}

    for conf in ["East", "West"]:
        r1_series  = bracket[conf]["R1"]
        r1_winners = []
        r1_details = []

        for s in r1_series:
            home, away = s["home"], s["away"]
            winner, win_pct, prob_home, game_dist = _predict_series(home, away, teams, bundle)
            loser = away if winner == home else home
            r1_winners.append(winner)
            r1_details.append({
                "home": home, "away": away,
                "home_name": teams[home]["name"],
                "away_name": teams[away]["name"],
                "home_seed": teams[home]["seed"],
                "away_seed": teams[away]["seed"],
                "winner": winner,
                "winner_name": teams[winner]["name"],
                "loser": loser,
                "win_pct": win_pct,
                "prob_home": prob_home,
                "game_dist": game_dist,
                "is_upset": teams[winner]["seed"] > teams[loser]["seed"],
            })

        results[conf]["R1"] = r1_details

        # R2: 1v4 winner vs 2v3 winner
        r2_pairs   = [(r1_winners[0], r1_winners[3]), (r1_winners[1], r1_winners[2])]
        r2_winners = []
        r2_details = []
        for t1, t2 in r2_pairs:
            home = t1 if teams[t1]["seed"] < teams[t2]["seed"] else t2
            away = t2 if home == t1 else t1
            winner, win_pct, prob_home, game_dist = _predict_series(home, away, teams, bundle)
            loser = away if winner == home else home
            r2_winners.append(winner)
            r2_details.append({
                "home": home, "away": away,
                "home_name": teams[home]["name"],
                "away_name": teams[away]["name"],
                "home_seed": teams[home]["seed"],
                "away_seed": teams[away]["seed"],
                "winner": winner,
                "winner_name": teams[winner]["name"],
                "loser": loser,
                "win_pct": win_pct,
                "prob_home": prob_home,
                "game_dist": game_dist,
                "is_upset": teams[winner]["seed"] > teams[loser]["seed"],
            })
        results[conf]["R2"] = r2_details

        # Conference Final
        h = r2_winners[0] if teams[r2_winners[0]]["seed"] < teams[r2_winners[1]]["seed"] else r2_winners[1]
        a = r2_winners[1] if h == r2_winners[0] else r2_winners[0]
        winner, win_pct, prob_home, game_dist = _predict_series(h, a, teams, bundle)
        loser = a if winner == h else h
        results[conf]["CF"] = {
            "home": h, "away": a,
            "home_name": teams[h]["name"], "away_name": teams[a]["name"],
            "home_seed": teams[h]["seed"], "away_seed": teams[a]["seed"],
            "winner": winner, "winner_name": teams[winner]["name"],
            "loser": loser,
            "win_pct": win_pct, "prob_home": prob_home,
            "game_dist": game_dist,
            "is_upset": teams[winner]["seed"] > teams[loser]["seed"],
        }
        results[conf]["champion"] = winner

    # NBA Finals
    east_champ = results["East"]["champion"]
    west_champ = results["West"]["champion"]
    h = east_champ if teams[east_champ]["W"] >= teams[west_champ]["W"] else west_champ
    a = west_champ if h == east_champ else east_champ
    winner, win_pct, prob_home, game_dist = _predict_series(h, a, teams, bundle)
    loser = a if winner == h else h
    results["Finals"] = {
        "home": h, "away": a,
        "home_name": teams[h]["name"], "away_name": teams[a]["name"],
        "home_seed": teams[h]["seed"], "away_seed": teams[a]["seed"],
        "winner": winner, "winner_name": teams[winner]["name"],
        "loser": loser,
        "win_pct": win_pct, "prob_home": prob_home,
        "game_dist": game_dist,
        "is_upset": teams[winner]["seed"] > teams[loser]["seed"],
        "champion": winner,
        "champion_name": teams[winner]["name"],
    }

    # Real results for comparison
    results["real_results"] = {
        "champion": "OKC",
        "champion_name": "Oklahoma City Thunder",
        "east_finalist": "CLE",
        "west_finalist": "OKC",
        "r1_upsets": [
            {"winner": "LAL", "loser": "LAC", "series": "LAC vs LAL"},
            {"winner": "GSW", "loser": "DEN", "series": "DEN vs GSW"},
            {"winner": "IND", "loser": "MIL", "series": "IND vs MIL"},
        ],
    }

    return results


# ── HTML page view ────────────────────────────────────────────────────────────
@login_required(login_url="/login/")
def playoff_page(request):
    return render(request, "predictor/playoffs.html")


# ── API: full bracket simulation ──────────────────────────────────────────────
@method_decorator(csrf_exempt, name="dispatch")
class PlayoffSimulateView(APIView):
    def get(self, request):
        if playoff_bundle is None:
            return Response(
                {"error": "Playoff model not loaded. Run: python train_playoff_model.py"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        if not teams_data:
            return Response(
                {"error": "Teams data not loaded. Check teams_stats_24_25.json"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        teams  = teams_data["teams"]
        sim    = _simulate_full_bracket(teams)

        # Add team metadata for the frontend
        teams_meta = {
            code: {
                "name":       t["name"],
                "code":       code,
                "conf":       t["conf"],
                "seed":       t["seed"],
                "W":          t["W"],
                "L":          t["L"],
                "net_rtg":    t["net_rtg"],
                "off_rtg":    t["off_rtg"],
                "def_rtg":    t["def_rtg"],
                "W_pct":      t["W_pct"],
                "playoff_exp":t["playoff_exp"],
            }
            for code, t in teams.items()
        }

        return Response({
            "season":     "2024-25",
            "simulation": sim,
            "teams":      teams_meta,
            "model_info": {
                "cv_accuracy": round(playoff_bundle["cv_accuracy"] * 100, 1),
                "cv_auc":      round(playoff_bundle["cv_auc"], 3),
                "training_series": 70,
                "features": playoff_bundle["features"],
            }
        })


# ── API: predict single series ────────────────────────────────────────────────
@method_decorator(csrf_exempt, name="dispatch")
class PlayoffSeriesView(APIView):
    def get(self, request, home_code, away_code):
        if playoff_bundle is None:
            return Response({"error": "Playoff model not loaded."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        teams = teams_data.get("teams", {})
        home_code = home_code.upper()
        away_code = away_code.upper()

        if home_code not in teams:
            return Response({"error": f"Team '{home_code}' not found."}, status=status.HTTP_404_NOT_FOUND)
        if away_code not in teams:
            return Response({"error": f"Team '{away_code}' not found."}, status=status.HTTP_404_NOT_FOUND)

        winner, win_pct, prob_home, game_dist = _predict_series(home_code, away_code, teams, playoff_bundle)

        return Response({
            "home":          home_code,
            "away":          away_code,
            "home_name":     teams[home_code]["name"],
            "away_name":     teams[away_code]["name"],
            "winner":        winner,
            "winner_name":   teams[winner]["name"],
            "win_pct":       win_pct,
            "prob_home":     prob_home,
            "game_dist":     game_dist,
            "is_upset":      teams[winner]["seed"] > teams[away_code if winner == home_code else home_code]["seed"],
        })


# ══════════════════════════════════════════════════════════════════════════════
# FANTASY VIEWS — append to bottom of views.py
# ══════════════════════════════════════════════════════════════════════════════
# Also add this import at the top of views.py:
#   from .models import FantasyLeague, FantasyTeam, FantasyRoster

FANTASY_SCORING = {
    "PTS":  1.0,
    "AST":  1.5,
    "TRB":  1.2,
    "STL":  2.0,
    "BLK":  2.0,
    "TOV": -1.0,
    "FG%":  10.0,
    "3P%":  8.0,
    "FT%":  5.0,
    "MP":   0.1,
}
BUDGET_CAP  = 100_000_000   # $100 million
MAX_PLAYERS = 5             # 5 starters only


def _compute_fantasy_score(player_row: dict) -> float:
    """Weighted fantasy score from per-game stats."""
    score = 0.0
    score += (player_row.get("PTS", 0) or 0) * FANTASY_SCORING["PTS"]
    score += (player_row.get("AST", 0) or 0) * FANTASY_SCORING["AST"]
    score += (player_row.get("TRB", 0) or 0) * FANTASY_SCORING["TRB"]
    score += (player_row.get("STL", 0) or 0) * FANTASY_SCORING["STL"]
    score += (player_row.get("BLK", 0) or 0) * FANTASY_SCORING["BLK"]
    score += (player_row.get("TOV", 0) or 0) * FANTASY_SCORING["TOV"]
    score += (player_row.get("FG%", 0) or 0) * FANTASY_SCORING["FG%"]
    score += (player_row.get("3P%", 0) or 0) * FANTASY_SCORING["3P%"]
    score += (player_row.get("FT%", 0) or 0) * FANTASY_SCORING["FT%"]
    score += (player_row.get("MP",  0) or 0) * FANTASY_SCORING["MP"]
    return round(max(score, 0), 2)


def _salary_from_score(score: float, min_score: float, max_score: float) -> int:
    """Map fantasy score to salary between $1M and $40M."""
    if max_score <= min_score:
        return 5_000_000
    pct = (score - min_score) / (max_score - min_score)
    return int((1 + pct * 39) * 1_000_000)


def _player_tier(score: float) -> str:
    if score >= 35: return "Superstar"
    if score >= 25: return "Star"
    if score >= 15: return "Starter"
    if score >=  8: return "Rotation"
    return "Bench"


def _get_or_create_league():
    """Get or create the global fantasy league."""
    from .models import FantasyLeague
    league, _ = FantasyLeague.objects.get_or_create(
        season="2024-25",
        defaults={"name": "NBA Fantasy 2024-25", "budget_cap": BUDGET_CAP, "max_players": 5}
    )
    return league


def _enrich_players():
    """Return df_avg rows enriched with fantasy_score and salary."""
    if df_avg is None:
        return []
    records = df_avg.copy()

    # Compute scores
    scores = []
    for _, row in records.iterrows():
        s = _compute_fantasy_score(row.to_dict())
        scores.append(s)
    records["fantasy_score"] = scores

    min_s = min(scores)
    max_s = max(scores)
    records["salary"] = records["fantasy_score"].apply(
        lambda s: _salary_from_score(s, min_s, max_s)
    )
    records["tier"] = records["fantasy_score"].apply(_player_tier)
    return records


# Build enriched player cache once at startup
_player_cache = None

def _get_player_cache():
    global _player_cache
    if _player_cache is None and df_avg is not None:
        _player_cache = _enrich_players()
    return _player_cache


# ── Fantasy diagnostic endpoint ──────────────────────────────────────────────
@csrf_exempt
def fantasy_debug(request):
    """GET /api/fantasy/debug/ — Check DB tables and auth status."""
    from django.db import connection
    tables = connection.introspection.table_names()
    fantasy_tables = [t for t in tables if 'fantasy' in t.lower()]
    return JsonResponse({
        "user":              str(request.user),
        "authenticated":     request.user.is_authenticated,
        "fantasy_tables":    fantasy_tables,
        "tables_exist": {
            "fantasyleague": "predictor_fantasyleague" in tables,
            "fantasyteam":   "predictor_fantasyteam"   in tables,
            "fantasyroster": "predictor_fantasyroster"  in tables,
        },
        "session_key":       request.session.session_key,
    })


# ── HTML page ─────────────────────────────────────────────────────────────────
@login_required(login_url="/login/")
def fantasy_page(request):
    return render(request, "predictor/fantasy.html")


# ── API: list all available players ──────────────────────────────────────────
@method_decorator(csrf_exempt, name="dispatch")
class FantasyPlayersView(APIView):
    def get(self, request):
        cache = _get_player_cache()
        if cache is None:
            return Response({"error": "Dataset not loaded."}, status=503)

        q          = request.GET.get("q", "").lower().strip()
        pos        = request.GET.get("pos", "").upper().strip()
        team       = request.GET.get("team", "").upper().strip()
        tier_f     = request.GET.get("tier", "").strip()
        sort_by    = request.GET.get("sort", "fantasy_score")
        page       = int(request.GET.get("page", 1))
        per_page   = int(request.GET.get("per_page", 40))
        max_salary = request.GET.get("max_salary", None)  # budget filter

        rows = cache.copy()

        if q:
            rows = rows[rows["Player"].str.lower().str.contains(q, na=False)]
        if pos:
            rows = rows[rows["Pos"] == pos]
        if team:
            rows = rows[rows["Tm"] == team]
        if tier_f:
            rows = rows[rows["tier"] == tier_f]
        if max_salary:
            try:
                rows = rows[rows["salary"] <= int(max_salary)]
            except (ValueError, TypeError):
                pass

        # Exclude players already on the requesting user's roster
        if request.user.is_authenticated:
            try:
                from .models import FantasyTeam
                league = _get_or_create_league()
                user_team = FantasyTeam.objects.get(user=request.user, league=league)
                on_roster = set(user_team.roster.values_list("player_name", flat=True))
                if on_roster:
                    rows = rows[~rows["Player"].isin(on_roster)]
            except Exception:
                pass

        valid_sorts = ["fantasy_score", "salary", "PTS", "AST", "TRB", "STL", "BLK", "MP"]
        if sort_by not in valid_sorts:
            sort_by = "fantasy_score"
        rows = rows.sort_values(sort_by, ascending=False)

        total   = len(rows)
        start   = (page - 1) * per_page
        page_df = rows.iloc[start:start + per_page]

        def clean(v):
            if v is None: return None
            try:
                if isinstance(v, float) and (v != v): return None
            except: pass
            return v

        players = []
        for _, row in page_df.iterrows():
            players.append({
                "name":          row["Player"],
                "team":          row["Tm"],
                "pos":           clean(row.get("Pos")),
                "age":           clean(row.get("Age")),
                "G":             clean(row.get("G")),
                "MP":            clean(row.get("MP")),
                "PTS":           clean(row.get("PTS")),
                "AST":           clean(row.get("AST")),
                "TRB":           clean(row.get("TRB")),
                "STL":           clean(row.get("STL")),
                "BLK":           clean(row.get("BLK")),
                "TOV":           clean(row.get("TOV")),
                "FG%":           clean(row.get("FG%")),
                "3P%":           clean(row.get("3P%")),
                "FT%":           clean(row.get("FT%")),
                "fantasy_score": round(float(row["fantasy_score"]), 2),
                "salary":        int(row["salary"]),
                "tier":          row["tier"],
            })

        return Response({
            "players":   players,
            "total":     total,
            "page":      page,
            "per_page":  per_page,
            "pages":     (total + per_page - 1) // per_page,
            "scoring":   FANTASY_SCORING,
        })


# ── API: get my team ──────────────────────────────────────────────────────────
@method_decorator(csrf_exempt, name="dispatch")
class FantasyMyTeamView(APIView):
    def get(self, request):
        if not request.user.is_authenticated:
            return Response({"error": "Login required."}, status=401)

        from .models import FantasyTeam, FantasyRoster
        league = _get_or_create_league()

        try:
            team = FantasyTeam.objects.get(user=request.user, league=league)
        except FantasyTeam.DoesNotExist:
            return Response({
                "exists":       False,
                "budget_cap":   BUDGET_CAP,
                "max_players":  MAX_PLAYERS,
                "scoring":      FANTASY_SCORING,
            })

        roster = []
        for entry in team.roster.all():
            roster.append({
                "player_name":   entry.player_name,
                "team_code":     entry.team_code,
                "position":      entry.position,
                "salary":        entry.salary,
                "fantasy_score": entry.fantasy_score,
                "PTS":  entry.pts,  "AST": entry.ast,
                "TRB":  entry.reb,  "STL": entry.stl,
                "BLK":  entry.blk,  "TOV": entry.tov,
                "FG%":  entry.fg_pct, "3P%": entry.three_pct,
                "FT%":  entry.ft_pct, "MP":  entry.mp,
            })

        budget_left = BUDGET_CAP - team.total_spent
        return Response({
            "exists":        True,
            "team_name":     team.name,
            "total_score":   team.total_score,
            "total_spent":   team.total_spent,
            "budget_left":   budget_left,
            "budget_cap":    BUDGET_CAP,
            "max_players":   MAX_PLAYERS,
            "roster_count":  len(roster),
            "roster":        roster,
            "scoring":       FANTASY_SCORING,
            "rank":          _get_team_rank(team),
        })

    def post(self, request):
        """Create or rename team."""
        if not request.user.is_authenticated:
            return Response({"error": "Login required."}, status=401)

        from .models import FantasyTeam
        league    = _get_or_create_league()
        team_name = request.data.get("team_name", "").strip()

        if not team_name:
            return Response({"error": "Team name required."}, status=400)
        if len(team_name) > 80:
            return Response({"error": "Team name too long (max 80 chars)."}, status=400)

        team, created = FantasyTeam.objects.get_or_create(
            user=request.user, league=league,
            defaults={"name": team_name, "total_score": 0, "total_spent": 0}
        )
        if not created:
            team.name = team_name
            team.save()

        return Response({
            "ok":      True,
            "created": created,
            "team_name": team.name,
        }, status=201 if created else 200)


# ── API: add player to roster ─────────────────────────────────────────────────
@method_decorator(csrf_exempt, name="dispatch")
class FantasyAddPlayerView(APIView):
    def post(self, request):
        if not request.user.is_authenticated:
            return Response({"error": "Login required."}, status=401)

        from .models import FantasyTeam, FantasyRoster
        league      = _get_or_create_league()
        player_name = request.data.get("player_name", "").strip()

        if not player_name:
            return Response({"error": "player_name required."}, status=400)

        # Get team
        try:
            team = FantasyTeam.objects.get(user=request.user, league=league)
        except FantasyTeam.DoesNotExist:
            return Response({"error": "Create your team first."}, status=400)

        # Check roster size
        if team.roster.count() >= MAX_PLAYERS:
            return Response({"error": "Roster full. You already have 5 starters (one per position)."}, status=400)

        # Already on roster?
        if team.roster.filter(player_name=player_name).exists():
            return Response({"error": f"{player_name} is already on your roster."}, status=409)

        # Enforce position rules: 5 starters, one per position

        # Find player in dataset
        cache = _get_player_cache()
        if cache is None:
            return Response({"error": "Dataset not loaded."}, status=503)

        p_rows = cache[cache["Player"] == player_name]
        if p_rows.empty:
            # Try case-insensitive
            p_rows = cache[cache["Player"].str.lower() == player_name.lower()]
        if p_rows.empty:
            return Response({"error": f"Player '{player_name}' not found."}, status=404)

        row  = p_rows.iloc[0]
        sal  = int(row["salary"])
        fscore = float(row["fantasy_score"])

        # Budget check
        if team.total_spent + sal > BUDGET_CAP:
            remaining = BUDGET_CAP - team.total_spent
            return Response({
                "error": f"Insufficient budget. Needs ${sal/1e6:.1f}M, you have ${remaining/1e6:.1f}M left."
            }, status=400)

        def safe(v, default=0.0):
            try:
                f = float(v)
                return 0.0 if f != f else f
            except: return default

        # Assign position — must match an open starter slot
        _pos_raw = str(row.get("Pos") or "").split("-")[0].strip().upper()
        _valid   = ["PG", "SG", "SF", "PF", "C"]
        if _pos_raw not in _valid:
            _pos_raw = "PG"   # fallback for unknown positions
        if team.roster.filter(position=_pos_raw).exists():
            return Response({
                "error": f"Position {_pos_raw} is already filled. Drop that player first."
            }, status=400)
        _final_pos = _pos_raw

        FantasyRoster.objects.create(
            team          = team,
            player_name   = player_name,
            team_code     = str(row.get("Tm", "")),
            position      = _final_pos,
            salary        = sal,
            fantasy_score = fscore,
            pts  = safe(row.get("PTS")),
            ast  = safe(row.get("AST")),
            reb  = safe(row.get("TRB")),
            stl  = safe(row.get("STL")),
            blk  = safe(row.get("BLK")),
            tov  = safe(row.get("TOV")),
            fg_pct   = safe(row.get("FG%")),
            three_pct= safe(row.get("3P%")),
            ft_pct   = safe(row.get("FT%")),
            mp   = safe(row.get("MP")),
        )

        team.recalculate_score()

        return Response({
            "ok":            True,
            "player_name":   player_name,
            "fantasy_score": fscore,
            "salary":        sal,
            "total_score":   team.total_score,
            "total_spent":   team.total_spent,
            "budget_left":   BUDGET_CAP - team.total_spent,
            "roster_count":  team.roster.count(),
        }, status=201)


# ── API: drop player from roster ─────────────────────────────────────────────
@method_decorator(csrf_exempt, name="dispatch")
class FantasyDropPlayerView(APIView):
    def post(self, request):
        if not request.user.is_authenticated:
            return Response({"error": "Login required."}, status=401)

        from .models import FantasyTeam, FantasyRoster
        league      = _get_or_create_league()
        player_name = request.data.get("player_name", "").strip()

        try:
            team = FantasyTeam.objects.get(user=request.user, league=league)
        except FantasyTeam.DoesNotExist:
            return Response({"error": "Team not found."}, status=404)

        deleted, _ = FantasyRoster.objects.filter(team=team, player_name=player_name).delete()
        if deleted == 0:
            return Response({"error": f"{player_name} not on your roster."}, status=404)

        team.recalculate_score()

        return Response({
            "ok":           True,
            "dropped":      player_name,
            "total_score":  team.total_score,
            "total_spent":  team.total_spent,
            "budget_left":  BUDGET_CAP - team.total_spent,
            "roster_count": team.roster.count(),
        })


# ── API: leaderboard ──────────────────────────────────────────────────────────
@method_decorator(csrf_exempt, name="dispatch")
class FantasyLeaderboardView(APIView):
    def get(self, request):
        from .models import FantasyTeam
        league = _get_or_create_league()
        teams  = FantasyTeam.objects.filter(league=league).order_by("-total_score")

        board = []
        for rank, team in enumerate(teams, 1):
            board.append({
                "rank":          rank,
                "team_name":     team.name,
                "username":      team.user.username,
                "full_name":     team.user.get_full_name(),
                "total_score":   team.total_score,
                "total_spent":   team.total_spent,
                "roster_count":  team.roster.count(),
                "is_me":         team.user == request.user if request.user.is_authenticated else False,
            })

        return Response({
            "leaderboard":  board,
            "total_teams":  len(board),
            "budget_cap":   BUDGET_CAP,
            "max_players":  MAX_PLAYERS,
        })


# ── Helper: get rank of a team ────────────────────────────────────────────────
def _get_team_rank(team):
    from .models import FantasyTeam
    league = team.league
    rank   = FantasyTeam.objects.filter(
        league=league, total_score__gt=team.total_score
    ).count() + 1
    return rank