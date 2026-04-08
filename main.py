import os
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

class MezallaEngine:
    def __init__(self):
        # GitHub Secrets - Bosluklari temizleyen .strip() eklendi
        self.sb_url = os.getenv('SB_URL', "").strip().rstrip("/")
        self.sb_key = os.getenv('SB_KEY', "").strip()
        self.tg_token = os.getenv('TG_TOKEN', "").strip()
        self.tg_chat_id = os.getenv('TG_CHAT_ID', "").strip()
        self.rapid_api_key = os.getenv('ODDS_API_KEY', "").strip()
        
        self.sb_headers = {
            "apikey": self.sb_key,
            "Authorization": f"Bearer {self.sb_key}",
            "Content-Type": "application/json"
        }
        
        self.bankroll = 585.60
        self.current_round = None
        self.next_kickoff = None
        self.model = None
        self.df = None
        self.features = [
            'rolling_xG', 'rolling_threat', 'position', 
            'next_difficulty', 'next_is_home', 'opp_leakage', 
            'fatigue_index', 'is_specialist', 'opp_defensive_form', 'avg_mins'
        ]

    def fetch_market_odds(self):
        """RapidAPI uzerinden piyasa oranlarini otonom ceker."""
        print(f"[{datetime.now().strftime('%H:%M')}] RapidAPI uzerinden oranlar cekiliyor...")
        if not self.rapid_api_key:
            print("Kritik Hata: ODDS_API_KEY bulunamadi.")
            return {}
            
        url = "https://the-odds-api.p.rapidapi.com/v4/sports/soccer_england_premier_league/odds/"
        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": "the-odds-api.p.rapidapi.com"
        }
        params = {
            'regions': 'eu',
            'markets': 'player_anytime_goalscorer',
            'oddsFormat': 'decimal'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code != 200:
                print(f"API Hatasi ({response.status_code}): {response.text}")
                return {}
                
            data = response.json()
            market_map = {}
            for match in data:
                for bookie in match.get('bookmakers', []):
                    if bookie['key'] in ['pinnacle', 'betfair_ex_back', 'williamhill', 'betfair_ex_lay']:
                        for market in bookie.get('markets', []):
                            if market['key'] == 'player_anytime_goalscorer':
                                for outcome in market['outcomes']:
                                    # En yuksek orani (en avantajli fiyati) kaydet
                                    current_max = market_map.get(outcome['name'], 0)
                                    if outcome['price'] > current_max:
                                        market_map[outcome['name']] = outcome['price']
            return market_map
        except Exception as e:
            print(f"RapidAPI Baglanti Hatasi: {e}")
            return {}

    def send_notification(self, message):
        """Mezalla Raporunu iletir."""
        if not self.tg_token or not self.tg_chat_id: return
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.tg_chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print(f"Telegram Hatasi: {e}")

    def fetch_and_process(self, limit=400):
        print(f"[{datetime.now().strftime('%H:%M')}] FPL verileri toplaniyor...")
        try:
            static_data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
            next_event = next(e for e in static_data['events'] if e['is_next'])
            self.current_round = next_event['id']
            self.next_kickoff = next_event['deadline_time']
            
            players = [p for p in static_data['elements'] if p['element_type'] in [3, 4]]
            all_match_data = []
            
            for p in players[:limit]:
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
                time.sleep(0.01)
            
            self.df = pd.DataFrame(all_match_data).sort_values(['player_id', 'round'])
            self._feature_engine()
        except Exception as e:
            print(f"Veri Isleme Hatasi: {e}")

    def _feature_engine(self):
        df = self.df
        df['avg_mins'] = df.groupby('player_id')['minutes'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['fatigue_index'] = df.groupby('player_id')['minutes'].transform(lambda x: x.rolling(2).sum())
        df['is_specialist'] = (df['threat'] > df['threat'].quantile(0.9)).astype(int)
        df['opp_leakage'] = df.groupby('difficulty')['goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['opp_defensive_form'] = df['opp_leakage']
        df['rolling_xG'] = df.groupby('player_id')['xG'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['rolling_threat'] = df.groupby('player_id')['threat'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['next_difficulty'] = df.groupby('player_id')['difficulty'].shift(-1)
        df['next_is_home'] = df.groupby('player_id')['is_home'].shift(-1)
        df['target'] = (df.groupby('player_id')['goals'].shift(-1) > 0).astype(int)
        self.df = df.dropna(subset=['target', 'next_difficulty'])

    def train_ensemble(self):
        print(f"[{datetime.now().strftime('%H:%M')}] Model egitiliyor...")
        X, y = self.df[self.features], self.df['target']
        ensemble = VotingClassifier(estimators=[
            ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss')),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=6))
        ], voting='soft')
        self.model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
        self.model.fit(X, y)

    def run_forecast_cycle(self):
        market_odds = self.fetch_market_odds()
        if not market_odds:
            print("Piyasa verisi bos dondu. Islem durduruldu.")
            return

        latest = self.df.groupby('player_id').tail(1).copy()
        latest['raw_prob'] = self.model.predict_proba(latest[self.features])[:, 1]
        latest['final_prob'] = latest['raw_prob'] * (0.8 + 0.2 * (latest['avg_mins']/90))
        
        latest['real_odds'] = latest['player_name'].map(market_odds)
        latest = latest.dropna(subset=['real_odds'])
        
        latest['ev'] = (latest['final_prob'] * latest['real_odds']) - 1
        signals = latest[latest['ev'] > 0.18].sort_values('ev', ascending=False).copy()
        signals['bet_amount'] = np.round(np.minimum(self.bankroll * 0.02, 25.0), 2)
        
        if not signals.empty:
            msg = f"🚀 *Mezalla Otonom Rapor (GW: {self.current_round})*\n\n"
            upsert_headers = self.sb_headers.copy()
            upsert_headers["Prefer"] = "resolution=merge-duplicates"

            for _, row in signals.iterrows():
                payload = {
                    "player_name": row['player_name'], "round": int(self.current_round),
                    "kickoff_time": self.next_kickoff, "prob": float(row['final_prob']),
                    "ev": float(row['ev']), "bet_amount": float(row['bet_amount']),
                    "status": "AUTO", "created_at": datetime.now().isoformat()
                }
                requests.post(f"{self.sb_url}/rest/v1/signals", headers=upsert_headers, json=payload, timeout=5)
                msg += f"👤 *{row['player_name']}*\n🎯 Olasilik: %{row['final_prob']*100:.1f}\n📈 EV: {row['ev']:.2f}\n💰 Oran: {row['real_odds']:.2f}\n💵 Bahis: {row['bet_amount']} TL\n\n"
            self.send_notification(msg)
        else:
            print("Kriterlere uygun firsat bulunamadi.")

if __name__ == "__main__":
    engine = MezallaEngine()
    engine.fetch_and_process()
    engine.train_ensemble()
    engine.run_forecast_cycle()
