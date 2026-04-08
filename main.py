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

    def fetch_current_fixtures(self):
        """Gelecek maçları ve rakipleri eşleştirir."""
        print(f"[{datetime.now().strftime('%H:%M')}] Fikstür haritası oluşturuluyor...")
        f_data = requests.get("https://fantasy.premierleague.com/api/fixtures/?future=1").json()
        static = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        team_map = {t['id']: t['name'] for t in static['teams']}
        
        fixture_map = {}
        for f in f_data:
            home = team_map[f['team_h']]
            away = team_map[f['team_a']]
            # Her iki takım için de bu maçın bilgisini tut
            fixture_map[home] = {"fixture_id": f['id'], "opponent": away, "is_home": True}
            fixture_map[away] = {"fixture_id": f['id'], "opponent": home, "is_home": False}
        return fixture_map

    def check_existing_prediction(self, fixture_id, team_name):
        """Aynı maç ve takım için kayıt var mı kontrol eder."""
        url = f"{self.sb_url}/rest/v1/predictions?fixture_id=eq.{fixture_id}&team_name=eq.{team_name}"
        res = requests.get(url, headers=self.headers).json()
        return len(res) > 0

    def fetch_fpl_data(self):
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        df_players = pd.DataFrame(r['elements'])
        df_teams = pd.DataFrame(r['teams'])[['id', 'name', 'strength']]
        df_teams.columns = ['team_id', 'team_name', 'team_strength']
        
        for col in ['expected_goals', 'threat']:
            df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0)
        
        team_agg = df_players.groupby('team').agg({'expected_goals': 'mean', 'threat': 'mean'}).reset_index()
        team_agg.columns = ['team_id', 'avg_xg', 'avg_threat']
        return pd.merge(team_agg, df_teams, on='team_id')

    def run_prediction_cycle(self, stats):
        print(f"[{datetime.now().strftime('%H:%M')}] Tahmin döngüsü başladı...")
        f_map = self.fetch_current_fixtures()
        
        # ML Model Eğitimi (Statik Veriyle)
        X = stats[self.features]
        y = (X['avg_xg'] > X['avg_xg'].median()).astype(int)
        model = CalibratedClassifierCV(VotingClassifier(estimators=[('xgb', XGBClassifier(n_estimators=100)), ('rf', RandomForestClassifier())], voting='soft'), cv=3).fit(X, y)

        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {"X-RapidAPI-Key": self.rapid_api_key, "X-RapidAPI-Host": "odds.p.rapidapi.com"}
        market_data = requests.get(url, headers=headers, params={'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}).json()

        for match in market_data:
            home_team = match['home_team']
            team_info = f_map.get(home_team)
            
            if not team_info: continue
            
            # Mükerrer Kayıt Kontrolü
            if self.check_existing_prediction(team_info['fixture_id'], home_team):
                print(f"⏩ Atlanıyor: {home_team} (Zaten kayıtlı)")
                continue

            team_row = stats[stats['team_name'].str.contains(home_team[:5], case=False)]
            if team_row.empty: continue
            
            prob = model.predict_proba(team_row[self.features])[0][1]
            best_odds = max([o['price'] for b in match['bookmakers'] for m in b['markets'] if m['key'] == 'h2h' for o in m['outcomes'] if o['name'] == home_team] or [0])
            ev = (prob * best_odds) - 1

            if ev > 0.05:
                bet = round(self.bankroll * 0.02, 2)
                payload = {
                    "fixture_id": team_info['fixture_id'],
                    "team_name": home_team, 
                    "model_version": "V5.7-PRO", 
                    "prob_home": float(prob), 
                    "ev_value": float(ev), 
                    "placed_odds": float(best_odds), 
                    "bet_amount": float(bet),
                    "avg_xg_snapshot": float(team_row['avg_xg'].iloc[0])
                }
                requests.post(f"{self.sb_url}/rest/v1/predictions", headers=self.headers, json=payload)
                self.send_telegram(home_team, team_info['opponent'], prob, best_odds, ev, bet)

    def send_telegram(self, team, opponent, prob, odds, ev, bet):
        msg = (f"🛡️ *Mezalla Enterprise v5.7*\n\n"
               f"🏟️ Maç: *{team} vs {opponent}*\n"
               f"🎯 Tahmin: {team}\n"
               f"📊 Olasılık: %{prob*100:.1f}\n"
               f"💰 Oran: {odds:.2f}\n"
               f"📈 EV: %{ev*100:.1f}\n"
               f"💵 Bahis: {bet} TL")
        requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", 
                      json={"chat_id": self.tg_chat_id, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    engine = MezallaEnterprise()
    stats = engine.fetch_fpl_data()
    engine.run_prediction_cycle(stats)
