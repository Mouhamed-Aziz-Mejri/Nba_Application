"""
fetch_player_info.py
--------------------
Fetches Age and Position for all NBA players in the dataset
from the Basketball Reference 2024-25 season stats page.

Run once:
    python manage.py fetch_player_info

Saves result to: predictor/data/player_info.json
"""

import json
import os
import time
import pandas as pd
from django.core.management.base import BaseCommand

# Try to use requests; fallback message if not installed
try:
    import requests
    from bs4 import BeautifulSoup
    DEPS_OK = True
except ImportError:
    DEPS_OK = False


DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
CACHE_FILE = os.path.join(DATA_DIR, "player_info.json")
CSV_PATH   = os.path.join(DATA_DIR, "database_24_25.csv")

# Basketball Reference 2024-25 per-game stats (no API key needed)
BREF_URL = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class Command(BaseCommand):
    help = "Fetch player Age and Position from Basketball Reference and cache to player_info.json"

    def handle(self, *args, **options):
        if not DEPS_OK:
            self.stderr.write(
                "❌ Missing dependencies. Install them first:\n"
                "   pip install requests beautifulsoup4 lxml\n"
            )
            return

        os.makedirs(DATA_DIR, exist_ok=True)

        # ── Load existing cache ───────────────────────────────────────────
        cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            self.stdout.write(f"📂 Loaded existing cache: {len(cache)} players")

        # ── Get player names from dataset ─────────────────────────────────
        if not os.path.exists(CSV_PATH):
            self.stderr.write(f"❌ Dataset not found at {CSV_PATH}")
            return

        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()

        player_col = "Player" if "Player" in df.columns else df.columns[0]
        players_in_dataset = df[player_col].dropna().unique().tolist()

        missing = [p for p in players_in_dataset if p not in cache]
        self.stdout.write(f"👥 {len(players_in_dataset)} players in dataset, {len(missing)} need fetching")

        if not missing:
            self.stdout.write("✅ All players already cached!")
            self._print_sample(cache)
            return

        # ── Scrape Basketball Reference ───────────────────────────────────
        self.stdout.write(f"🌐 Fetching from Basketball Reference...")

        try:
            resp = requests.get(BREF_URL, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            self.stderr.write(f"❌ Failed to fetch Basketball Reference: {e}")
            self._fallback_nba_api(missing, cache)
            return

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", {"id": "per_game_stats"})

        if not table:
            self.stderr.write("❌ Could not find stats table on Basketball Reference")
            self._fallback_nba_api(missing, cache)
            return

        fetched = 0
        rows = table.find("tbody").find_all("tr")

        for row in rows:
            # Skip header rows
            if row.get("class") and "thead" in row.get("class"):
                continue

            name_td = row.find("td", {"data-stat": "name_display"})
            pos_td  = row.find("td", {"data-stat": "pos"})
            age_td  = row.find("td", {"data-stat": "age"})

            if not name_td:
                continue

            # Clean player name (remove special chars Basketball Ref adds)
            name = name_td.get_text(strip=True).replace("*", "").strip()
            pos  = pos_td.get_text(strip=True)  if pos_td  else ""
            age  = age_td.get_text(strip=True)  if age_td  else ""

            if name and name not in cache:
                cache[name] = {
                    "Pos": pos.split("-")[0].strip() if pos else None,
                    "Age": int(age) if age.isdigit() else None,
                }
                fetched += 1

        self.stdout.write(f"✅ Fetched {fetched} new players from Basketball Reference")

        # ── Save cache ────────────────────────────────────────────────────
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

        self.stdout.write(f"💾 Saved to {CACHE_FILE} ({len(cache)} total players)")
        self._print_sample(cache)

        # Report still missing
        still_missing = [p for p in players_in_dataset if p not in cache]
        if still_missing:
            self.stdout.write(
                f"\n⚠️  {len(still_missing)} players not found on Basketball Reference:\n"
                + "\n".join(f"   - {p}" for p in still_missing[:10])
            )

    def _fallback_nba_api(self, missing, cache):
        """Fallback: try nba_api library if installed."""
        try:
            from nba_api.stats.endpoints import commonplayerinfo
            from nba_api.stats.static import players as nba_players

            self.stdout.write("🔄 Trying nba_api as fallback...")

            all_players = nba_players.get_players()
            name_map    = {p["full_name"]: p["id"] for p in all_players}

            for name in missing:
                player_id = name_map.get(name)
                if not player_id:
                    continue
                try:
                    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
                    df_info = info.get_data_frames()[0]
                    cache[name] = {
                        "Pos": str(df_info["POSITION"].iloc[0]).split("-")[0].strip() or None,
                        "Age": int(df_info["SEASON_EXP"].iloc[0]) + 18 if pd.notna(df_info["SEASON_EXP"].iloc[0]) else None,
                    }
                    time.sleep(0.6)  # respect rate limit
                except Exception:
                    pass

            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=2)
            self.stdout.write(f"💾 nba_api fallback saved {len(cache)} players")

        except ImportError:
            self.stdout.write(
                "⚠️  nba_api not installed. Install with: pip install nba_api\n"
                "   Then re-run: python manage.py fetch_player_info"
            )

    def _print_sample(self, cache):
        self.stdout.write("\n📋 Sample:")
        for name, info in list(cache.items())[:5]:
            self.stdout.write(f"   {name}: Age={info.get('Age')}, Pos={info.get('Pos')}")