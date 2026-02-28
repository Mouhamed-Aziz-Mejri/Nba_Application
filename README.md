# 🏀 NBA Player Performance — Django REST API

---

## Project Structure

```
nba_project/
├── manage.py
├── requirements.txt
├── nba_project/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── predictor/
    ├── views.py        ← prediction logic
    ├── urls.py         ← API routes
    ├── apps.py
    └── model/
        └── nba_model.pkl   ← place your model here
```

---

## Setup & Run

### 1. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place your model
Copy `nba_model.pkl` into:
```
predictor/model/nba_model.pkl
```

### 4. Run migrations & start server
```bash
python manage.py migrate
python manage.py runserver
```

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/health/` | Check server & model status |
| POST | `/api/predict/` | Predict player performance |

---

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "LeBron James",
    "age": 28,
    "games_played": 75,
    "minutes_per_game": 34.5,
    "points_per_game": 25.3,
    "assists_per_game": 7.2,
    "rebounds_per_game": 8.1,
    "steals_per_game": 1.3,
    "blocks_per_game": 0.8,
    "turnovers_per_game": 3.1,
    "field_goal_pct": 0.512,
    "three_point_pct": 0.364,
    "free_throw_pct": 0.731,
    "plus_minus": 5.2
  }'
```

## Example Response

```json
{
  "player_name": "LeBron James",
  "performance_score": 78.45,
  "performance_tier": "All-Star",
  "message": "LeBron James is projected as an All-Star-level performer with a score of 78.45."
}
```

---

## Performance Tiers

| Score | Tier |
|-------|------|
| ≥ 80 | Elite |
| 65–79 | All-Star |
| 50–64 | Starter |
| 35–49 | Rotation |
| < 35  | Bench |
