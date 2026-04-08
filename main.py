import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

class MezallaEnterprise:
    def __init__(self):
        self.sb_url = os.getenv('SB_URL', "").strip().rstrip("/")
        self.sb_key = os.getenv('SB_KEY', "").strip()
        self.tg_token = os.getenv('TG_TOKEN', "").strip()
        self.tg_chat_id = os.getenv('TG_CHAT_ID', "").strip()
        self.rapid_api_key = os.getenv('ODDS_API_KEY', "").strip()
        
        self.headers = {
            "apikey": self.sb_key,
            "Authorization": f"Bearer {self.sb_key}",
            "Content-Type": "application/json"
        }
        self.bankroll = 585.60
        self.features = ['strength_diff', 'xg_diff', 'threat_diff']
        
        # --- KRITIK: Takim Isim Esleme Sozlugu ---
        self.name_map = {
            "Manchester City": "Man City",
            "Manchester United": "Man Utd",
            "Tottenham Hotspur": "Spurs",
            "Wolverhampton Wanderers": "Wolves",
            "Newcastle United": "Newcastle",
            "Sheffield United": "Sheffield Utd",
            "Nottingham Forest": "Nott'm Forest",
            "Brighton and Hove Albion": "Brighton",
            "Luton Town": "Luton",
            "West Ham United": "West Ham"
        }

    def normalize_name(self, name):
        """Odds API ismini FPL formatina donusturur."""
        return self.name_map.get(name, name)

    def fetch_fixture_map(self):
        try:
            f_data = requests.get("https://fantasy.premierleague.com/api/fixtures/?future=1").json()
            static = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
            team_map = {t['id']: t['name'] for t in static['teams']}
            
            fixture_map = {}
            for f in f_data:
                h_team, a_team = team_map[f['team_h']], team_map[f['team_a']]
                fixture_map[h_team] = {"fixture_id": f['id'], "opponent": a_team}
                fixture_map[a_team] = {"fixture_id": f['id'], "opponent": h_team}
            return fixture_map
        except: return {}

    def fetch_team_stats(self):
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        df_players = pd.DataFrame(r['elements'])
        df_teams = pd.DataFrame(r['teams'])[['id', 'name', 'strength']]
        
        for col in ['expected_goals', 'threat']:
            df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0)
            
        team_agg = df_players.groupby('team').agg({'expected_goals': 'mean', 'threat': 'mean'}).reset_index()
        return pd.merge(team_agg, df_teams, left_on='team', right_on='id').drop('id', axis=1)

    def run_engine(self):
        print(f"[{datetime.now().strftime('%H:%M')}] Mezalla V6.0 Operasyonu Basladi")
        f_map = self.fetch_fixture_map()
        stats = self.fetch_team_stats()
        
        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {"X-RapidAPI-Key": self.rapid_api_key, "X-RapidAPI-Host": "odds.p.rapidapi.com"}
        try:
            market_data = requests.get(url, headers=headers, params={'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}).json()
        except: return

        for match in market_data:
            # Isimleri normalize et
            raw_h, raw_a = match['home_team'], match['away_team']
            norm_h, norm_a = self.normalize_name(raw_h), self.normalize_name(raw_a)
            
            f_info = f_map.get(norm_h)
            if not f_info: continue

            # Takim verilerini cek
            h_row = stats[stats['name'] == norm_h]
            a_row = stats[stats['name'] == norm_a]

            if h_row.empty or a_row.empty:
                print(f"⚠️ Atlandi: Eslestirme Yapilamadi ({norm_h} vs {norm_a})")
                continue

            h_s, a_s = h_row.iloc[0], a_row.iloc[0]

            # --- ML Olasilik Tahmini (Gelistirilmis) ---
            strength_diff = h_s['strength'] - a_s['strength']
            xg_diff = h_s['expected_goals'] - a_s['expected_goals']
            
            # Basit ama rasyonel bir olasilik modeli
            # Ev sahibi avantaji + Guc farki + Form (xG) farki
            prob = 0.38 + (strength_diff * 0.05) + (xg_diff * 0.15)
            prob = np.clip(prob, 0.15, 0.82) # %82 uzeri olasilik risklidir, kirptik.

            # Oran Analizi
            best_odds = 0
            for bookie in match['bookmakers']:
                for m in bookie['markets']:
                    if m['key'] == 'h2h':
                        for o in m['outcomes']:
                            if o['name'] == raw_h:
                                best_odds = max(best_odds, o['price'])

            if best_odds <= 1.0: continue
            ev = (prob * best_odds) - 1

            # Rasyonel EV Filtresi (%3 ile %45 arasi)
            if 0.03 < ev < 0.45:
                # Mukerrer Kontrol
                check = requests.get(f"{self.sb_url}/rest/v1/predictions?fixture_id=eq.{f_info['fixture_id']}&team_name=eq.{norm_h}", headers=self.headers).json()
                if check: continue

                bet = round(self.bankroll * 0.02, 2)
                payload = {
                    "fixture_id": f_info['fixture_id'],
                    "team_name": norm_h,
                    "model_version": "V6.0-NORMALIZED",
                    "prob_home": float(prob),
                    "ev_value": float(ev),
                    "placed_odds": float(best_odds),
                    "bet_amount": float(bet)
                }
                requests.post(f"{self.sb_url}/rest/v1/predictions", headers=self.headers, json=payload)
                self.send_telegram(norm_h, norm_a, prob, best_odds, ev, bet)

    def send_telegram(self, h, a, p, o, ev, b):
        msg = (f"🛡️ *Mezalla Enterprise v6.0*\n\n"
               f"🏟️ Mac: {h} vs {a}\n"
               f"🎯 Tahmin: {h}\n"
               f"📊 Olasılık: %{p*100:.1f}\n"
               f"💰 Oran: {o:.2f}\n"
               f"📈 EV: %{ev*100:.1f}\n"
               f"💵 Bahis: {b} TL")
        requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", 
                      json={"chat_id": self.tg_chat_id, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    engine = MezallaEnterprise()
    engine.run_engine()
