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
        self.features = ['team_strength', 'avg_xg', 'avg_threat']

    def fetch_fixture_map(self):
        """Gelecek maçları ID ve rakip bilgisiyle haritalar."""
        f_data = requests.get("https://fantasy.premierleague.com/api/fixtures/?future=1").json()
        static = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        team_map = {t['id']: t['name'] for t in static['teams']}
        
        fixture_map = {}
        for f in f_data:
            h_team, a_team = team_map[f['team_h']], team_map[f['team_a']]
            fixture_map[h_team] = {"fixture_id": f['id'], "opponent": a_team}
            fixture_map[a_team] = {"fixture_id": f['id'], "opponent": h_team}
        return fixture_map

    def check_duplicate(self, f_id, team):
        """Aynı maç ve takım için kayıt kontrolü yapar."""
        url = f"{self.sb_url}/rest/v1/predictions?fixture_id=eq.{f_id}&team_name=eq.{team}"
        res = requests.get(url, headers=self.headers).json()
        return len(res) > 0

    def fetch_fpl_stats(self):
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        df_players = pd.DataFrame(r['elements'])
        df_teams = pd.DataFrame(r['teams'])[['id', 'name', 'strength']]
        df_teams.columns = ['team_id', 'team_name', 'team_strength']
        
        for col in ['expected_goals', 'threat']:
            df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0)
        
        team_agg = df_players.groupby('team').agg({'expected_goals': 'mean', 'threat': 'mean'}).reset_index()
        team_agg.columns = ['team_id', 'avg_xg', 'avg_threat']
        return pd.merge(team_agg, df_teams, on='team_id')

    def run_engine(self):
        print(f"[{datetime.now().strftime('%H:%M')}] Mezalla V5.8 Operasyonu Basladı...")
        fixture_map = self.fetch_fixture_map()
        stats = self.fetch_fpl_stats()
        
        # Model Eğitimi
        X = stats[self.features]
        y = (X['avg_xg'] > X['avg_xg'].median()).astype(int)
        model = CalibratedClassifierCV(VotingClassifier(estimators=[('xgb', XGBClassifier(n_estimators=100)), ('rf', RandomForestClassifier())], voting='soft'), cv=3).fit(X, y)

        # Oran Çekimi
        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {"X-RapidAPI-Key": self.rapid_api_key, "X-RapidAPI-Host": "odds.p.rapidapi.com"}
        market_data = requests.get(url, headers=headers, params={'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}).json()

        for match in market_data:
            home_team = match['home_team']
            f_info = fixture_map.get(home_team)
            
            if not f_info or self.check_duplicate(f_info['fixture_id'], home_team):
                continue

            team_row = stats[stats['team_name'].str.contains(home_team[:5], case=False)]
            if team_row.empty: continue
            
            prob = model.predict_proba(team_row[self.features])[0][1]
            best_odds = max([o['price'] for b in match['bookmakers'] for m in b['markets'] if m['key'] == 'h2h' for o in m['outcomes'] if o['name'] == home_team] or [0])
            ev = (prob * best_odds) - 1

            if ev > 0.10:
                bet = round(self.bankroll * 0.02, 2)
                payload = {
                    "fixture_id": f_info['fixture_id'],
                    "team_name": home_team, 
                    "model_version": "V5.8-RELATIONAL",
                    "prob_home": float(prob),
                    "ev_value": float(ev),
                    "placed_odds": float(best_odds),
                    "bet_amount": float(bet),
                    "avg_xg_snapshot": float(team_row['avg_xg'].iloc[0])
                }
                requests.post(f"{self.sb_url}/rest/v1/predictions", headers=self.headers, json=payload)
                self.send_telegram(home_team, f_info['opponent'], prob, best_odds, ev, bet)

    def send_telegram(self, team, opponent, prob, odds, ev, bet):
        msg = (f"Mezalla Enterprise v5.8\n\n"
               f"Mac: {team} vs {opponent}\n"
               f"Tahmin: {team}\n"
               f"Olasılık: %{prob*100:.1f}\n"
               f"Oran: {odds:.2f}\n"
               f"EV: %{ev*100:.1f}\n"
               f"Bahis: {bet} TL")
        requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", json={"chat_id": self.tg_chat_id, "text": msg})

if __name__ == "__main__":
    engine = MezallaEnterprise()
    engine.run_engine()
