"""
train_playoff_model.py
----------------------
Trains an XGBoost classifier to predict NBA playoff series winners.

Features: team stat differentials (net rating, offense, defense,
pace, win %, experience, rest) + seed difference.

Run from project root:
    python train_playoff_model.py

Outputs:
    predictor/model/playoff_model.pkl   ← trained model bundle
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH  = os.path.join(BASE_DIR, "predictor", "data", "playoff_series_history.csv")
TEAMS_PATH    = os.path.join(BASE_DIR, "predictor", "data", "teams_stats_24_25.json")
MODEL_OUT     = os.path.join(BASE_DIR, "predictor", "model", "playoff_model.pkl")

print("=" * 60)
print("  NBA Playoff Series Predictor — Training")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD & ENGINEER FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n📂 Loading historical series data...")
df = pd.read_csv(HISTORY_PATH)
print(f"   {len(df)} series loaded  (2015–2024)")
print(f"   Higher seed wins: {df['target'].mean():.1%}")

def build_features(row):
    """
    Convert raw team stats into DIFFERENTIAL features.
    All features are (home_team - away_team) so positive = home advantage.
    The model learns: 'how much better is the higher-seeded/home team?'
    """
    return {
        # Seed advantage (higher seed = lower number = better)
        "seed_diff":      row["away_seed"]  - row["home_seed"],    # positive = home is higher seed

        # Core team quality
        "net_rtg_diff":   row["home_net"]   - row["away_net"],
        "off_rtg_diff":   row["home_off"]   - row["away_off"],
        "def_rtg_diff":   row["away_def"]   - row["home_def"],     # flipped: lower def_rtg is better

        # Wins
        "W_diff":         row["home_W"]     - row["away_W"],

        # Playing style
        "pace_diff":      row["home_pace"]  - row["away_pace"],

        # Experience
        "exp_diff":       row["home_exp"]   - row["away_exp"],

        # Rest
        "rest_diff":      row["rest_diff"],

        # Interaction: net_rtg gap × seed gap (captures "dominant favorite" scenarios)
        "quality_seed_interaction": (row["home_net"] - row["away_net"]) * (row["away_seed"] - row["home_seed"]),

        # Offensive vs defensive matchup
        "off_vs_def":     row["home_off"]   - row["away_def"],     # home offense vs away defense
        "def_vs_off":     row["away_def"]   - row["home_off"],     # away defense vs home offense
    }

feature_rows = [build_features(row) for _, row in df.iterrows()]
X = pd.DataFrame(feature_rows)
y = df["target"].values

FEATURES = X.columns.tolist()
print(f"\n📐 Features engineered ({len(FEATURES)}):")
for f in FEATURES:
    print(f"   • {f}")

# ══════════════════════════════════════════════════════════════════
# 2. CROSS-VALIDATION — compare models
# ══════════════════════════════════════════════════════════════════
print("\n🔄 Running 5-fold stratified cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scale for logistic regression
scaler_cv = StandardScaler()
X_scaled  = scaler_cv.fit_transform(X)

models_cv = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "XGBoost":             XGBClassifier(
                               n_estimators=150, max_depth=3,
                               learning_rate=0.08, subsample=0.8,
                               colsample_bytree=0.8, min_child_weight=3,
                               use_label_encoder=False, eval_metric="logloss",
                               random_state=42, verbosity=0
                           ),
}

cv_results = {}
print(f"\n{'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10}")
print("-" * 48)
for name, clf in models_cv.items():
    X_in = X_scaled if "Logistic" in name else X
    acc  = cross_val_score(clf, X_in, y, cv=cv, scoring="accuracy").mean()
    auc  = cross_val_score(clf, X_in, y, cv=cv, scoring="roc_auc").mean()
    cv_results[name] = {"accuracy": acc, "auc": auc}
    print(f"  {name:<23} {acc:>9.1%} {auc:>9.3f}")

# ══════════════════════════════════════════════════════════════════
# 3. TRAIN FINAL XGBoost MODEL ON ALL DATA
# ══════════════════════════════════════════════════════════════════
print("\n🤖 Training final XGBoost on full dataset...")

scaler = StandardScaler()
X_scaled_final = scaler.fit_transform(X)

final_model = XGBClassifier(
    n_estimators=150, max_depth=3,
    learning_rate=0.08, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, verbosity=0
)
final_model.fit(X, y)   # XGBoost uses raw features (handles scale internally)

# Full dataset accuracy (upper bound — shows model fit)
y_pred     = final_model.predict(X)
y_prob     = final_model.predict_proba(X)[:, 1]
train_acc  = accuracy_score(y, y_pred)
train_auc  = roc_auc_score(y, y_prob)

print(f"\n   Train accuracy : {train_acc:.1%}")
print(f"   Train AUC-ROC  : {train_auc:.3f}")
print(f"   CV accuracy    : {cv_results['XGBoost']['accuracy']:.1%}  ← honest estimate")
print(f"   CV AUC-ROC     : {cv_results['XGBoost']['auc']:.3f}")

# ── Confusion matrix ──────────────────────────────────────────────
cm = confusion_matrix(y, y_pred)
print(f"\n   Confusion matrix (train):")
print(f"   {'':>20} Pred: Upset  Pred: Favored")
print(f"   Actual Upset:    {cm[0][0]:>4}          {cm[0][1]:>4}")
print(f"   Actual Favored:  {cm[1][0]:>4}          {cm[1][1]:>4}")

# ── Feature importance ────────────────────────────────────────────
importances = final_model.feature_importances_
feat_imp    = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
print(f"\n📊 Feature importances:")
for feat, imp in feat_imp:
    bar = "█" * int(imp * 40)
    print(f"   {feat:<32} {imp:.3f}  {bar}")

# ══════════════════════════════════════════════════════════════════
# 4. VALIDATE ON 2024-25 SEASON (our target simulation)
# ══════════════════════════════════════════════════════════════════
print("\n\n🏀 Validating on 2024-25 First Round (real results)...")
print("-" * 60)

with open(TEAMS_PATH, encoding="utf-8") as f:
    teams_data = json.load(f)
teams = teams_data["teams"]

# Real 2024-25 first round results
# Format: (home_code, away_code, actual_winner, games_played)
real_r1 = [
    # EAST
    ("BOS", "CHI",  "BOS", 4),
    ("CLE", "MIA",  "CLE", 4),
    ("NYK", "ORL",  "NYK", 5),
    ("IND", "MIL",  "IND", 6),
    # WEST
    ("OKC", "MEM",  "OKC", 4),
    ("HOU", "MIN",  "HOU", 6),
    ("LAC", "LAL",  "LAL", 6),   # upset: LAL won
    ("DEN", "GSW",  "GSW", 7),   # upset: GSW won
]

def predict_series(home_code, away_code, teams, model, features_list):
    """Predict series winner and win probability."""
    h = teams[home_code]
    a = teams[away_code]
    row = {
        "home_seed": h["seed"], "away_seed": a["seed"],
        "home_net":  h["net_rtg"],   "away_net":  a["net_rtg"],
        "home_off":  h["off_rtg"],   "away_off":  a["off_rtg"],
        "home_def":  h["def_rtg"],   "away_def":  a["def_rtg"],
        "home_W":    h["W"],         "away_W":    a["W"],
        "home_pace": h["pace"],      "away_pace": a["pace"],
        "home_exp":  h["playoff_exp"],"away_exp": a["playoff_exp"],
        "rest_diff": 0,
    }
    feats = build_features(row)
    X_row = pd.DataFrame([feats])[features_list]
    prob_home = float(model.predict_proba(X_row)[0][1])
    predicted  = home_code if prob_home >= 0.5 else away_code
    prob_win   = prob_home if prob_home >= 0.5 else (1 - prob_home)
    return predicted, round(prob_win * 100, 1), round(prob_home * 100, 1)

correct = 0
predictions_r1 = []
for home, away, actual, games in real_r1:
    pred, prob_win, prob_home = predict_series(home, away, teams, final_model, FEATURES)
    is_correct = pred == actual
    correct   += int(is_correct)
    status     = "✅" if is_correct else "❌"
    h_name     = teams[home]["name"].split()[-1]
    a_name     = teams[away]["name"].split()[-1]
    print(f"  {status} {h_name:12s} vs {a_name:12s}  →  Predicted: {teams[pred]['name'].split()[-1]:12s}  ({prob_win:.0f}%)  Actual: {teams[actual]['name'].split()[-1]} in {games}")
    predictions_r1.append({
        "home": home, "away": away, "predicted": pred,
        "prob_home": prob_home, "actual": actual, "correct": is_correct
    })

print(f"\n  R1 Accuracy: {correct}/{len(real_r1)} = {correct/len(real_r1):.1%}")

# ── Simulate full 2024-25 bracket ─────────────────────────────────
print("\n\n🏆 Full 2024-25 Playoff Bracket Simulation:")
print("=" * 60)

bracket = {
    "East": {
        "R1": [("BOS","CHI"), ("CLE","MIA"), ("NYK","ORL"), ("IND","MIL")],
    },
    "West": {
        "R1": [("OKC","MEM"), ("HOU","MIN"), ("LAC","LAL"), ("DEN","GSW")],
    }
}

simulation_results = {"East": {}, "West": {}}

for conf in ["East", "West"]:
    print(f"\n  ── {conf}ern Conference ──")
    r1_winners = []
    print("  Round 1:")
    for home, away in bracket[conf]["R1"]:
        pred, prob, prob_home = predict_series(home, away, teams, final_model, FEATURES)
        loser = away if pred == home else home
        h_name = teams[home]["name"].split()[-1]
        a_name = teams[away]["name"].split()[-1]
        print(f"    {h_name} vs {a_name}  →  {teams[pred]['name'].split()[-1]} ({prob:.0f}%)")
        r1_winners.append(pred)
    simulation_results[conf]["R1"] = r1_winners

    # R2: 1v4 winner vs 2v3 winner  (bracket positions)
    r2_pairs = [(r1_winners[0], r1_winners[3]), (r1_winners[1], r1_winners[2])]
    r2_winners = []
    print("  Round 2 (Semifinals):")
    for t1, t2 in r2_pairs:
        # Higher seed hosts
        h = t1 if teams[t1]["seed"] < teams[t2]["seed"] else t2
        a = t2 if h == t1 else t1
        pred, prob, prob_home = predict_series(h, a, teams, final_model, FEATURES)
        print(f"    {teams[h]['name'].split()[-1]} vs {teams[a]['name'].split()[-1]}  →  {teams[pred]['name'].split()[-1]} ({prob:.0f}%)")
        r2_winners.append(pred)
    simulation_results[conf]["R2"] = r2_winners

    # Conference Final
    h = r2_winners[0] if teams[r2_winners[0]]["seed"] < teams[r2_winners[1]]["seed"] else r2_winners[1]
    a = r2_winners[1] if h == r2_winners[0] else r2_winners[0]
    conf_winner, prob, prob_home = predict_series(h, a, teams, final_model, FEATURES)
    print(f"  Conference Final:")
    print(f"    {teams[h]['name'].split()[-1]} vs {teams[a]['name'].split()[-1]}  →  {teams[conf_winner]['name'].split()[-1]} ({prob:.0f}%)")
    simulation_results[conf]["Final"] = conf_winner

# NBA Finals
east_champ = simulation_results["East"]["Final"]
west_champ = simulation_results["West"]["Final"]
# Higher seed (better record) gets home court
h = east_champ if teams[east_champ]["W"] >= teams[west_champ]["W"] else west_champ
a = west_champ if h == east_champ else east_champ
champion, prob, prob_home = predict_series(h, a, teams, final_model, FEATURES)

print(f"\n  🏆 NBA FINALS:")
print(f"    {teams[h]['name']} vs {teams[a]['name']}")
print(f"    Predicted Champion: {teams[champion]['name']} ({prob:.0f}%)")
print(f"\n  Real Champion: Oklahoma City Thunder ✅" if champion == "OKC" else f"\n  Real Champion: Oklahoma City Thunder")

# ══════════════════════════════════════════════════════════════════
# 5. SAVE MODEL BUNDLE
# ══════════════════════════════════════════════════════════════════
bundle = {
    "model":        final_model,
    "scaler":       scaler,
    "features":     FEATURES,
    "cv_accuracy":  cv_results["XGBoost"]["accuracy"],
    "cv_auc":       cv_results["XGBoost"]["auc"],
    "build_features_source": """
def build_features(row):
    return {
        'seed_diff':      row['away_seed']  - row['home_seed'],
        'net_rtg_diff':   row['home_net']   - row['away_net'],
        'off_rtg_diff':   row['home_off']   - row['away_off'],
        'def_rtg_diff':   row['away_def']   - row['home_def'],
        'W_diff':         row['home_W']     - row['away_W'],
        'pace_diff':      row['home_pace']  - row['away_pace'],
        'exp_diff':       row['home_exp']   - row['away_exp'],
        'rest_diff':      row['rest_diff'],
        'quality_seed_interaction': (row['home_net'] - row['away_net']) * (row['away_seed'] - row['home_seed']),
        'off_vs_def':     row['home_off']   - row['away_def'],
        'def_vs_off':     row['away_def']   - row['home_off'],
    }
""",
    "r1_validation": {
        "correct": correct,
        "total":   len(real_r1),
        "accuracy": correct / len(real_r1),
    },
    "simulation_2024_25": {
        "predicted_champion": champion,
        "real_champion":      "OKC",
        "correct":            champion == "OKC",
        "east_finalist":      east_champ,
        "west_finalist":      west_champ,
    },
}

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(bundle, MODEL_OUT)

print(f"\n\n✅ Model saved to {MODEL_OUT}")
print(f"   CV Accuracy : {cv_results['XGBoost']['accuracy']:.1%}")
print(f"   CV AUC-ROC  : {cv_results['XGBoost']['auc']:.3f}")
print(f"   R1 Accuracy : {correct}/{len(real_r1)} = {correct/len(real_r1):.1%}")
print(f"   Champion prediction correct: {champion == 'OKC'}")
print("\n🏀 Training complete!")