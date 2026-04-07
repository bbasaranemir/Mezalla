import os
import requests
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

class RegistaIntelligence:
    def __init__(self):
        # GitHub Secrets / Environment Variables
        self.sb_url = os.getenv('SB_URL', "").strip().rstrip("/")
        self.sb_key = os.getenv('SB_KEY', "")
        self.tg_token = os.getenv('TG_TOKEN', "")
        self.tg_chat_id = os.getenv('TG_CHAT_ID', "")
        
        self.headers = {
            "apikey": self.sb_key,
            "Authorization": f"Bearer {self.sb_key}",
            "Content-Type": "application/json"
        }
        
        self.bankroll = 585.60
        self.confidence_multiplier = 1.0
        self.model = None
        self.df = None
        self.synergy_matrix = {}
        self.features = [
            'rolling_xG', 'rolling_threat', 'position', 
            'next_difficulty', 'next_is_home', 'opp_leakage',
            'fatigue_index', 'is_specialist', 'opp_defensive_form', 'avg_mins'
        ]

    def send_notification(self, message):
        """Telegram uzerinden anlik rapor iletir."""
        if not self.tg_token or not self.tg_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        payload = {"chat_id": self.tg_chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"Telegram hatasi: {e}")

    def fetch_and_process(self, limit=400):
        """FPL API verilerini ceker ve ozellik muhendisligi yapar."""
        print(f"[{datetime.now().strftime('%H:%M')}] Veri toplama basladi...")
        try:
            raw = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
            players = [p for p in raw['elements'] if p['element_type'] in [3, 4]]
            
            all_match_data = []
            for i, p in enumerate(players[:limit]):
                p_id, p_name = p['id'], f"{p['first_name']} {p['second_name']}"
                try:
                    r = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{p_id}/", timeout=10).json()
                    for m in r.get('history', []):
                        all_match_data.append({
                            'player_id': p_id, 'player_name': p_name, 'team_id': p['team'],
                            'position': p['element_type'], 'round': m['round'],
                            'xG': float(m.get('expected_goals', 0)), 'threat': float(m.get('threat', 0)),
                            'minutes': m.get('minutes', 0), 'difficulty': m.get('difficulty', 3),
                            'goals': m.get('goals_scored', 0), 'is_home': 1 if m.get('was_home') else 0
                        })
                except: continue
                if (i+1) % 100 == 0: print(f"Okunan oyuncu: {i+1}")
                time.sleep(0.05)
            
            self.df = pd.DataFrame(all_match_data).sort_values(['player_id', 'round'])
            self._feature_engine()
        except Exception as e:
            print(f"Veri cekme hatasi: {e}")

    def _feature_engine(self):
        # Sinerji Haritasi
        for team in self.df['team_id'].unique():
            t_df = self.df[self.df['team_id'] == team]
            pivot = t_df.pivot_table(index='round', columns='player_name', values='xG', aggfunc='sum').fillna(0)
            corr = pivot.corr()
            for player in corr.columns:
                self.synergy_matrix[player] = corr[player].sort_values(ascending=False)[1:3].index.tolist()

        # Teknik Metrikler
        self.df['avg_mins'] = self.df.groupby('player_id')['minutes'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        self.df['fatigue_index'] = self.df.groupby('player_id')['minutes'].transform(lambda x: x.rolling(2).sum())
        self.df['is_specialist'] = (self.df['threat'] > self.df['threat'].quantile(0.9)).astype(int)
        self.df['opp_defensive_form'] = self.df.groupby('difficulty')['goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        self.df['opp_leakage'] = self.df.groupby('difficulty')['goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        self.df['rolling_xG'] = self.df.groupby('player_id')['xG'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        self.df['rolling_threat'] = self.df.groupby('player_id')['threat'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        self.df['next_difficulty'] = self.df.groupby('player_id')['difficulty'].shift(-1)
        self.df['next_is_home'] = self.df.groupby('player_id')['is_home'].shift(-1)
        self.df['target'] = (self.df.groupby('player_id')['goals'].shift(-1) > 0).astype(int)
        self.df = self.df.dropna(subset=['target', 'next_difficulty'])

    def train_ensemble(self):
        """Hibrit model egitimi ve kalibrasyon."""
        X, y = self.df[self.features], self.df['target']
        ensemble = VotingClassifier(estimators=[
            ('xgb', XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=5, eval_metric='logloss')),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=7))
        ], voting='soft')
        self.model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=5)
        self.model.fit(X, y)
        print("Model egitimi tamamlandi.")

    def sync_pnl_autonomous(self):
        """Supabase uzerinden performansi denetler."""
        if not self.sb_url: return
        print("Supabase kar/zarar analizi yapiliyor...")
        try:
            endpoint = f"{self.sb_url}/rest/v1/performance_analysis?select=*"
            r = requests.get(endpoint, headers=self.headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if not data: return
                pnl = sum([(d['bet_amount'] * (1/d['prob']) - d['bet_amount']) if d['hit'] else -d['bet_amount'] for d in data])
                self.bankroll += pnl
                acc = sum([d['hit'] for d in data]) / len(data)
                self.confidence_multiplier = 1.1 if acc > 0.6 else (0.85 if acc < 0.4 else 1.0)
                print(f"Guncel Kasa: {self.bankroll:.2f} TL")
        except:
            print("Supabase baglantisi kurulamadi, yerel verilerle devam ediliyor.")

    def run_forecast_cycle(self, market_odds):
        """Sinyalleri uretir, Supabase'e kaydeder ve Telegram'a gonderir."""
        self.sync_pnl_autonomous()
        latest = self.df.groupby('player_id').tail(1).copy()
        latest['raw_prob'] = self.model.predict_proba(latest[self.features])[:, 1]
        
        # Olasilik simulasyonu
        latest['final_prob'] = latest['raw_prob'] * (0.8 + 0.2 * (latest['avg_mins']/90))
        latest['real_odds'] = latest['player_name'].map(market_odds)
        latest = latest.dropna(subset=['real_odds'])
        latest['ev'] = (latest['final_prob'] * latest['real_odds']) - 1
        
        signals = latest[latest['ev'] > 0.18].sort_values('ev', ascending=False).copy()
        signals['bet_amount'] = np.round(np.minimum(self.bankroll * 0.02 * self.confidence_multiplier, 25.0), 2)
        
        if not signals.empty:
            msg = "🔔 *Regista Intelligence Raporu*\n\n"
            for _, row in signals.iterrows():
                payload = {
                    "player_name": row['player_name'], "prob": float(row['final_prob']),
                    "ev": float(row['ev']), "bet_amount": float(row['bet_amount']),
                    "status": "AUTO", "created_at": datetime.now().isoformat()
                }
                # Supabase Kayit
                if self.sb_url:
                    requests.post(f"{self.sb_url}/rest/v1/signals", headers=self.headers, json=payload, timeout=5)
                
                msg += f"👤 *{row['player_name']}*\n🎯 Olasilik: %{row['final_prob']*100:.1f}\n📈 EV: {row['ev']:.2f}\n💰 Bahis: {row['bet_amount']} TL\n\n"
            
            self.send_notification(msg)
        return signals

if __name__ == "__main__":
    engine = RegistaIntelligence()
    engine.fetch_and_process()
    engine.train_ensemble()
    
    # Ornek Piyasa Verisi (Gercek kullanimda bu veriler otomatik cekilebilir)
    market = {
        'Anthony Gordon': 3.40,
        'Danny Welbeck': 3.25,
        'Nicolas Jackson': 2.90,
        'Ollie Watkins': 2.10
    }
    
    report = engine.run_forecast_cycle(market)
    print(report[['player_name', 'final_prob', 'ev', 'bet_amount']])
