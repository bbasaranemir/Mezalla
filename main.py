import os
import requests
import pandas as pd
import numpy as np
import time
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
        self.features = ['team_strength', 'avg_xg', 'avg_threat', 'difficulty_index']
        self.model = None

    def fetch_fpl_data(self):
        """Hata Giderildi: Player ve Team verileri dogru sekilde birlestirildi."""
        print(f"[{datetime.now().strftime('%H:%M')}] FPL verileri toplaniyor...")
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        
        df_players = pd.DataFrame(r['elements'])
        df_teams = pd.DataFrame(r['teams'])[['id', 'name', 'strength']]
        df_teams.columns = ['team_id', 'team_name', 'team_strength']
        
        # Oyuncu bazli xG ve Threat metriklerini takim seviyesine indirge
        team_agg = df_players.groupby('team').agg({
            'expected_goals': 'mean',
            'threat': 'mean',
            'difficulty': 'mean'
        }).reset_index()
        team_agg.columns = ['team_id', 'avg_xg', 'avg_threat', 'difficulty_index']
        
        # Takim gucu (strength) ile birlestir
        full_stats = pd.merge(team_agg, df_teams, on='team_id')
        return full_stats

    def train_ml_model(self, stats):
        print(f"[{datetime.now().strftime('%H:%M')}] ML Modeli egitiliyor...")
        X = stats[self.features]
        # Hedef: Kendi ortalamasinin uzerinde xG ureten "formda" takimlari bulmak
        y = (X['avg_xg'] > X['avg_xg'].median()).astype(int)
        
        ensemble = VotingClassifier(estimators=[
            ('xgb', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5))
        ], voting='soft')
        
        self.model = CalibratedClassifierCV(ensemble, cv=3)
        self.model.fit(X, y)

    def log_to_supabase(self, table, data):
        """Profesyonel DB loglama."""
        try:
            url = f"{self.sb_url}/rest/v1/{table}"
            requests.post(url, headers=self.headers, json=data, timeout=10)
        except Exception as e:
            print(f"DB Log Hatasi ({table}): {e}")

    def fetch_odds(self):
        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {"X-RapidAPI-Key": self.rapid_api_key, "X-RapidAPI-Host": "odds.p.rapidapi.com"}
        params = {'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
        res = requests.get(url, headers=headers, params=params, timeout=15)
        return res.json() if res.status_code == 200 else []

    def run_engine(self):
        stats = self.fetch_fpl_data()
        self.train_ml_model(stats)
        market_data = self.fetch_odds()
        
        if not market_data:
            print("Piyasa verisi çekilemedi.")
            return

        for match in market_data:
            home_team_name = match['home_team']
            # Basit string esleme (Gelistirilebilir)
            team_row = stats[stats['team_name'].str.contains(home_team_name[:5], case=False)]
            
            if team_row.empty: continue
            
            # ML Tahmini
            prob_home = self.model.predict_proba(team_row[self.features])[0][1]
            
            # Oran Analizi
            best_odds = 0
            for bookie in match['bookmakers']:
                for market in bookie['markets']:
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team_name:
                            best_odds = max(best_odds, outcome['price'])

            ev = (prob_home * best_odds) - 1
            
            if ev > 0.03: # %3 ve uzeri avantaj yakalandiysa
                bet = np.round(self.bankroll * 0.02, 2)
                
                prediction_entry = {
                    "model_version": "V5.1-HYBRID",
                    "home_avg_xg": float(team_row['avg_xg'].iloc[0]),
                    "prob_home": float(prob_home),
                    "prediction_result": "HOME"
                }
                
                # Otonom Sinyal
                self.log_to_supabase("predictions", prediction_entry)
                self.send_telegram(home_team_name, prob_home, best_odds, ev, bet)

    def send_telegram(self, team, prob, odds, ev, bet):
        msg = (f"🤖 *Mezalla Enterprise v5.1*\n\n"
               f"⚽ Takim: {team}\n"
               f"📊 ML Olasilik: %{prob*100:.1f}\n"
               f"💰 En Iyi Oran: {odds:.2f}\n"
               f"📈 EV: %{ev*100:.1f}\n"
               f"💵 Onerilen: {bet} TL")
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        requests.post(url, json={"chat_id": self.tg_chat_id, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    engine = MezallaEnterprise()
    engine.run_engine()
    
