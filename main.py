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

    # --- DENETİM (AUDITOR) BÖLÜMÜ ---
    def audit_past_predictions(self):
        """PENDING tahminleri gerçek sonuçlarla kapatır."""
        print(f"[{datetime.now().strftime('%H:%M')}] Denetim döngüsü başlatıldı...")
        url = f"{self.sb_url}/rest/v1/predictions?actual_result=eq.PENDING"
        pending = requests.get(url, headers=self.headers).json()
        
        if not pending: return

        # FPL Sonuçlarını Çek
        fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
        static = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        team_map = {t['id']: t['name'] for t in static['teams']}
        
        results = {}
        for f in fixtures:
            if f['finished']:
                h_name = team_map[f['team_h']]
                res = "HOME_WIN" if f['team_h_score'] > f['team_a_score'] else ("AWAY_WIN" if f['team_h_score'] < f['team_a_score'] else "DRAW")
                results[h_name] = res

        for p in pending:
            p_id, team = p['id'], p['team_name']
            odds, bet = float(p.get('placed_odds', 1.0)), float(p.get('bet_amount', 0.0))
            
            actual, pl = "LOSS", -bet
            for h_team, res in results.items():
                if team in h_team:
                    if res == "HOME_WIN": actual, pl = "WIN", round((bet * odds) - bet, 2)
                    elif res == "DRAW": actual, pl = "DRAW", 0
                    break
            
            requests.patch(f"{self.sb_url}/rest/v1/predictions?id=eq.{p_id}", headers=self.headers, json={"actual_result": actual, "profit_loss": pl})
            print(f"✅ Audit: {team} -> {actual}")

    # --- ML VE TAHMİN BÖLÜMÜ ---
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
        print(f"[{datetime.now().strftime('%H:%M')}] Tahmin döngüsü başlatıldı...")
        # Model Eğitimi
        X = stats[self.features]
        y = (X['avg_xg'] > X['avg_xg'].median()).astype(int)
        ensemble = VotingClassifier(estimators=[('xgb', XGBClassifier(n_estimators=100, max_depth=4)), ('rf', RandomForestClassifier(n_estimators=100))], voting='soft')
        model = CalibratedClassifierCV(ensemble, cv=3).fit(X, y)

        # Oran Çekimi ve Analiz
        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {"X-RapidAPI-Key": self.rapid_api_key, "X-RapidAPI-Host": "odds.p.rapidapi.com"}
        market_data = requests.get(url, headers=headers, params={'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}).json()

        for match in market_data:
            home_team = match['home_team']
            team_row = stats[stats['team_name'].str.contains(home_team[:5], case=False)]
            if team_row.empty: continue
            
            prob = model.predict_proba(team_row[self.features])[0][1]
            best_odds = max([o['price'] for b in match['bookmakers'] for m in b['markets'] if m['key'] == 'h2h' for o in m['outcomes'] if o['name'] == home_team] or [0])
            ev = (prob * best_odds) - 1

            if ev > 0.05:
                bet = round(self.bankroll * 0.02, 2)
                payload = {"team_name": home_team, "model_version": "V5.6-INTEGRATED", "home_strength": int(team_row['team_strength'].iloc[0]), "avg_xg_snapshot": float(team_row['avg_xg'].iloc[0]), "avg_threat_snapshot": float(team_row['avg_threat'].iloc[0]), "prob_home": float(prob), "ev_value": float(ev), "placed_odds": float(best_odds), "bet_amount": float(bet)}
                requests.post(f"{self.sb_url}/rest/v1/predictions", headers=self.headers, json=payload)
                self.send_telegram(home_team, prob, best_odds, ev, bet)

    def send_telegram(self, team, prob, odds, ev, bet):
        msg = f"🛡️ *Mezalla Enterprise v5.6*\n\n⚽ Takım: {team}\n🎯 ML: %{prob*100:.1f}\n💰 Oran: {odds:.2f}\n📈 EV: %{ev*10
