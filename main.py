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
        except Exception as e:
            print(f"Fikstur verisi cekilemedi: {e}")
            return {}

    def fetch_team_stats(self):
        try:
            r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
            df_players = pd.DataFrame(r['elements'])
            df_teams = pd.DataFrame(r['teams'])[['id', 'name', 'strength']]
            
            for col in ['expected_goals', 'threat']:
                df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0)
                
            team_agg = df_players.groupby('team').agg({'expected_goals': 'mean', 'threat': 'mean'}).reset_index()
            stats = pd.merge(team_agg, df_teams, left_on='team', right_on='id')
            return stats.drop('id', axis=1)
        except Exception as e:
            print(f"Istatistik verisi cekilemedi: {e}")
            return pd.DataFrame()

    def run_engine(self):
        print(f"[{datetime.now().strftime('%H:%M')}] Mezalla V5.9.1 Operasyonu Basladi")
        fixture_map = self.fetch_fixture_map()
        stats = self.fetch_team_stats()
        
        if stats.empty: return

        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {"X-RapidAPI-Key": self.rapid_api_key, "X-RapidAPI-Host": "odds.p.rapidapi.com"}
        try:
            market_data = requests.get(url, headers=headers, params={'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}).json()
        except: return

        for match in market_data:
            home_team = match['home_team']
            away_team = match['away_team']
            f_info = fixture_map.get(home_team)

            if not f_info: continue

            # Takim eslestirme kontrolu (Hata onleme)
            h_row = stats[stats['name'].str.contains(home_team[:4], case=False)]
            a_row = stats[stats['name'].str.contains(away_team[:4], case=False)]

            if h_row.empty or a_row.empty:
                print(f"⚠️ Atlandi: Takim eslesmedi ({home_team} veya {away_team})")
                continue

            h_stats = h_row.iloc[0]
            a_stats = a_row.iloc[0]

            # Diferansiyel Hesaplama
            diff_strength = h_stats['strength'] - a_stats['strength']
            diff_xg = h_stats['expected_goals'] - a_stats['expected_goals']
            
            # Olasilik Tahmini (Rasyonel Sinirlar)
            base_prob = 0.35 
            prob = base_prob + (diff_strength * 0.04) + (diff_xg * 0.12)
            prob = np.clip(prob, 0.10, 0.85)

            # Oran Analizi
            best_odds = 0
            for bookie in match['bookmakers']:
                for market in bookie['markets']:
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                best_odds = max(best_odds, outcome['price'])

            if best_odds == 0: continue
            ev = (prob * best_odds) - 1

            # Rasyonel EV Filtresi
            if 0.03 < ev < 0.40:
                # Mükerrer Kayıt Kontrolü (DB sorgusu)
                check_url = f"{self.sb_url}/rest/v1/predictions?fixture_id=eq.{f_info['fixture_id']}&team_name=eq.{home_team}"
                if len(requests.get(check_url, headers=self.headers).json()) > 0:
                    continue

                bet = round(self.bankroll * 0.02, 2)
                payload = {
                    "fixture_id": f_info['fixture_id'],
                    "team_name": home_team,
                    "model_version": "V5.9.1-ROBUST",
                    "prob_home": float(prob),
                    "ev_value": float(ev),
                    "placed_odds": float(best_odds),
                    "bet_amount": float(bet)
                }
                requests.post(f"{self.sb_url}/rest/v1/predictions", headers=self.headers, json=payload)
                self.send_telegram(home_team, away_team, prob, best_odds, ev, bet)

    def send_telegram(self, home, away, prob, odds, ev, bet):
        msg = (f"Mezalla Enterprise v5.9.1\n\n"
               f"Mac: {home} vs {away}\n"
               f"Tahmin: {home}\n"
               f"Olasilik: %{prob*100:.1f}\n"
               f"Oran: {odds:.2f}\n"
               f"EV: %{ev*100:.1f}\n"
               f"Bahis: {bet} TL")
        requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", json={"chat_id": self.tg_chat_id, "text": msg})

if __name__ == "__main__":
    engine = MezallaEnterprise()
    engine.run_engine()
