import os
import requests
from datetime import datetime, date

API_URL = "https://api.football-data.org/v4/matches"
API_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN", "")

def get_today_fixtures():
    # Mock mode if token missing or set to "MOCK"
    if not API_TOKEN or API_TOKEN.upper() == "MOCK":
        return {
            "ok": True,
            "matches": [
                {"time": "20:00", "home": "الأهلي", "away": "الزمالك", "competition": "دوري تجريبي"},
                {"time": "22:30", "home": "الهلال", "away": "النصر", "competition": "ودية"},
            ],
        }
    today = date.today().isoformat()
    headers = {"X-Auth-Token": API_TOKEN}
    params = {"dateFrom": today, "dateTo": today}
    try:
        r = requests.get(API_URL, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        matches = data.get("matches", [])
        local_tz = datetime.now().astimezone().tzinfo
        out = []
        for m in matches:
            utc = m.get("utcDate") or ""
            try:
                dt_local = datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone(local_tz)
                t = dt_local.strftime("%H:%M")
            except Exception:
                t = utc
            home = (m.get("homeTeam") or {}).get("name", "")
            away = (m.get("awayTeam") or {}).get("name", "")
            comp = (m.get("competition") or {}).get("name", "")
            out.append({"time": t, "home": home, "away": away, "competition": comp})
        return {"ok": True, "matches": out}
    except Exception as e:
        return {"ok": False, "reason": str(e)}