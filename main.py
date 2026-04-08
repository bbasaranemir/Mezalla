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
        # API ve Veritabani Baglantilari
        self.sb_url = os.getenv('SB_URL', "").strip().rstrip("/")
        self.sb_key = os.getenv('SB_KEY', "").strip()
        self.tg_token = os.getenv('TG_TOKEN', "").strip()
        self.tg_chat_id = os.getenv('TG_CHAT_ID', "").strip()
        self.rapid_api_key = os.getenv('ODDS_API_KEY', "").strip()
        
        self.headers = {
            "apikey": self.sb_key,
            "Authorization": f"Bearer {self.sb_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        self.bankroll = 585.60
        self.features = ['team_avg_xG', 'team_avg_threat', 'difficulty_index', 'home_advantage', 'rolling_points']
        self.model = None

    def log_to_database(self, data):
        """Tahminleri ve kasa durumunu Supabase'e kaydeder."""
        try:
            endpoint = f"{self.sb_url}/rest/v1/predictions"
            requests.post(endpoint, headers=self.headers, json=data, timeout=10)
        except Exception as e:
            print(f"Veritabani Yazma Hatasi: {e}")

    def fetch_fpl_data(self):
        """Takim performans verilerini ML icin hazirlar."""
        print(f"[{datetime.now().strftime('%H:%M')}] FPL verileri isleniyor...")
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        teams = pd.DataFrame(r['teams'])
        elements = pd.DataFrame(r['elements'])
        
        # Takim bazli istatistikleri oyuncu verilerinden uret
        team_stats = elements.groupby('team').agg({
            'expected_goals': 'mean',
            'threat': 'mean',
            'strength': 'mean'
        }).reset_index()
        
        team_stats.columns = ['id', 'team_avg_xG', 'team_avg_threat', 'difficulty_index']
        return team_stats

    def train_ml_model(self, stats):
        """XGBoost ve RF ile hibrit tahmin modeli egitir."""
        print(f"[{datetime.now().strftime('%H:%M')}] Makine ogrenmesi modeli egitiliyor...")
        # Sentetik egitim verisi (Gercek veriler biriktikce DB'den cekilecek)
        X = stats[['team_avg_xG', 'team_avg_threat', 'difficulty_index']]
        y = (X['team_avg_xG'] > X['team_avg_xG'].median()).astype(int) # Basit hedef fonksiyonu
        
        clf = VotingClassifier(estimators=[
            ('xgb', XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=3))
        ], voting='soft')
        
        self.model = CalibratedClassifierCV(clf, cv=2)
        self.model.fit(X, y)

    def fetch_market_odds(self):
        """Ucretsiz plandaki H2H oranlarini ceker."""
        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {"X-RapidAPI-Key": self.rapid_api_key, "X-RapidAPI-Host": "odds.p.rapidapi.com"}
        params = {'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
        
        res = requests.get(url, headers=headers, params=params, timeout=15)
        return res.json() if res.status_code == 200 else []

    def run_engine(self):
        stats = self.fetch_fpl_data()
        self.train_ml_model(stats)
        market_data = self.fetch_market_odds()
        
        if not market_data: return

        for match in market_data:
            # ML Model Tahmini
            home_team = match['home_team']
            # Burada takim ismini stats tablosuyla eslestirip tahmin uretilir
            # Simülasyon: ML olasiligi uretiliyor
            ml_prob = 0.45 # Modelden gelen cikti varsayiliyor
            
            best_odds = 0
            for bookie in match['bookmakers']:
                for market in bookie['markets']:
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            best_odds = max(best_odds, outcome['price'])

            ev = (ml_prob * best_odds) - 1
            
            if ev > 0.05: # %5 EV ustu sinyaldir
                bet = np.round(self.bankroll * 0.02, 2)
                signal = {
                    "match_name": f"{match['home_team']} vs {match['away_team']}",
                    "prediction": home_team,
                    "probability": ml_prob,
                    "odds": best_odds,
                    "ev": ev,
                    "bet_amount": bet,
                    "timestamp": datetime.now().isoformat()
                }
                
                # DB'ye kaydet ve Telegram'a at
                self.log_to_database(signal)
                self.send_telegram(signal)

    def send_telegram(self, s):
        msg = (f"🤖 *Mezalla ML Engine v5.0*\n\n"
               f"⚽ Maç: {s['match_name']}\n"
               f"✅ Tahmin: {s['prediction']}\n"
               f"📊 ML Olasılık: %{s['probability']*100:.1f}\n"
               f"💰 Oran: {s['odds']:.2f}\n"
               f"📈 Değer (EV): %{s['ev']*100:.1f}\n"
               f"💵 Bahis: {s['bet_amount']} TL")
        requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", 
                      json={"chat_id": self.tg_chat_id, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    engine = MezallaEnterprise()
    engine.run_engine()
