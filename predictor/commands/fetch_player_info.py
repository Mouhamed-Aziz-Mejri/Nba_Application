"""
fetch_player_info.py
--------------------
Fetches Age and Position for all NBA players in ONE single API call
using the NBA stats leaguedashplayerstats endpoint.

Run once:
    python manage.py fetch_player_info

Saves to: predictor/data/player_info.json
"""

import json
import os
import pandas as pd
from django.core.management.base import BaseCommand

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
CACHE_FILE = os.path.join(DATA_DIR, "player_info.json")
CSV_PATH   = os.path.join(DATA_DIR, "database_24_25.csv")

# One single request — returns all ~500 players at once
NBA_STATS_URL = (
    "https://stats.nba.com/stats/leaguedashplayerstats"
    "?College=&Conference=&Country=&DateFrom=&DateTo=&Division="
    "&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&ISTRound="
    "&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0"
    "&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame"
    "&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N"
    "&Season=2024-25&SeasonSegment=&SeasonType=Regular+Season"
    "&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference="
    "&VsDivision=&Weight="
)

HEADERS = {
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "en-US,en;q=0.9",
    "Connection":       "keep-alive",
    "Host":             "stats.nba.com",
    "Origin":           "https://www.nba.com",
    "Referer":          "https://www.nba.com/",
    "User-Agent":       (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
}


class Command(BaseCommand):
    help = "Fetch all player Age+Position in one request from NBA stats API"

    def handle(self, *args, **options):
        try:
            import requests
        except ImportError:
            self.stderr.write("❌ Run: pip install requests")
            return

        os.makedirs(DATA_DIR, exist_ok=True)

        # ── Load existing cache ───────────────────────────────────────────
        cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            self.stdout.write(f"📂 Existing cache: {len(cache)} players")

        # ── ONE single request to get all players ─────────────────────────
        self.stdout.write("🌐 Fetching all players in one request from stats.nba.com...")

        try:
            resp = requests.get(
                NBA_STATS_URL,
                headers=HEADERS,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            self.stderr.write(f"❌ Request failed: {e}")
            self._try_csv_fallback(cache)
            return

        # ── Parse response ────────────────────────────────────────────────
        try:
            result_set = data["resultSets"][0]
            headers_list = result_set["headers"]
            rows         = result_set["rowSet"]

            # Find column indices
            name_idx = headers_list.index("PLAYER_NAME")
            age_idx  = headers_list.index("AGE")        if "AGE"  in headers_list else None
            pos_idx  = headers_list.index("PLAYER_POSITION") if "PLAYER_POSITION" in headers_list else None

            self.stdout.write(f"📋 Headers: {headers_list[:10]}...")
            self.stdout.write(f"   name_idx={name_idx}, age_idx={age_idx}, pos_idx={pos_idx}")
            self.stdout.write(f"   Total players: {len(rows)}")

            fetched = 0
            for row in rows:
                name = row[name_idx]
                if not name:
                    continue

                age = None
                pos = None

                if age_idx is not None and row[age_idx] is not None:
                    try:
                        age = int(float(row[age_idx]))
                    except (ValueError, TypeError):
                        pass

                if pos_idx is not None and row[pos_idx]:
                    pos = str(row[pos_idx]).split("-")[0].strip()

                cache[name] = {"Pos": pos, "Age": age}
                fetched += 1

            self.stdout.write(f"✅ Fetched {fetched} players!")

        except (KeyError, IndexError, ValueError) as e:
            self.stderr.write(f"❌ Failed to parse response: {e}")
            self.stderr.write(f"   Keys in response: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            return

        # ── Save cache ────────────────────────────────────────────────────
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

        self.stdout.write(f"💾 Saved to {CACHE_FILE} ({len(cache)} total players)")
        self._print_sample(cache)

        # ── Report coverage ───────────────────────────────────────────────
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df.columns = df.columns.str.strip()
            player_col = "Player" if "Player" in df.columns else df.columns[0]
            dataset_players = set(df[player_col].dropna().unique())
            covered = dataset_players & set(cache.keys())
            missing = dataset_players - set(cache.keys())
            self.stdout.write(f"\n📊 Coverage: {len(covered)}/{len(dataset_players)} players from your dataset")
            if missing:
                self.stdout.write(f"⚠️  {len(missing)} not matched (name differences):")
                for p in sorted(missing)[:15]:
                    self.stdout.write(f"   - {p}")

    def _try_csv_fallback(self, cache):
        """
        Last resort: if user has a separate NBA roster CSV,
        build the cache from that. Otherwise print instructions.
        """
        self.stdout.write(
            "\n💡 Manual fallback options:\n"
            "   1. Install requests and retry:  pip install requests\n"
            "   2. Try with a VPN if stats.nba.com is blocked in your region\n"
            "   3. Download this CSV manually and place it in predictor/data/nba_rosters.csv:\n"
            "      https://raw.githubusercontent.com/bttmly/nba/master/data/players.json\n"
        )

    def _print_sample(self, cache):
        self.stdout.write("\n📋 Sample:")
        for name, info in list(cache.items())[:8]:
            self.stdout.write(f"   {name:28s}  Age={info.get('Age')}  Pos={info.get('Pos')}")