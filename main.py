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
        # API ve DB Baglantilari
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
        
        # Isim Normalizasyon Sozlugu
        self.name_map = {
            "Manchester City": "Man City", "Manchester United": "Man Utd",
            "Tottenham Hotspur": "Spurs", "Wolverhampton Wanderers": "Wolves",
            "Newcastle United": "Newcastle", "Sheffield United": "Sheffield Utd",
            "Nottingham Forest": "Nott'm Forest", "Brighton and Hove Albion": "Brighton",
            "Luton Town": "Luton", "West Ham United": "West Ham"
        }

    def normalize_name(self, name):
        return self.name_map.get(name, name)

    def audit_past_predictions(self):
        """PENDING olan maclari ID bazli denetler ve kasayi gunceller."""
        print(f"[{datetime.now().strftime('%H:%M')}] Denetim basladi...")
        url = f"{self.sb_url}/rest/v1/predictions?actual_result=eq.PENDING"
        try:
            pending = requests.get(url, headers=self.headers, timeout=10).json()
            if not pending: 
                print("Bekleyen PENDING tahmini bulunamadi.")
                return

            fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
            f_results = {f['id']: f for f in fixtures if f['finished']}
            
            static = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
            t_map = {t['name']: t['id'] for t in static['teams']}

            for p in pending:
                p_id, f_id = p['id'], p.get('fixture_id')
                pred_team = p['team_name']
                odds, bet = float(p.get('placed_odds', 1.0)), float(p.get('bet_amount', 0.0))

                if f_id not in f_results: continue

                match = f_results[f_id]
                h_score, a_score = match['team_h_score'], match['team_a_score']
                norm_team = self.normalize_name(pred_team)
                pred_team_id = t_map.get(norm_team)

                actual, pl = "LOSS", -bet
                if (match['team_h'] == pred_team_id and h_score > a_score) or \
                   (match['team_a'] == pred_team_id and a_score > h_score):
                    actual, pl = "WIN", round((bet * odds) - bet, 2)
                elif h_score == a_score:
                    actual, pl = "DRAW", 0

                requests.patch(f"{self.sb_url}/rest/v1/predictions?id=eq.{p_id}", 
                               headers=self.headers, json={"actual_result": actual, "profit_loss": pl})
                print(f"✅ Audit: {pred_team} -> {actual}")
        except Exception as e:
            print(f"Audit Hatasi: {e}")

    def fetch_team_stats(self):
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        df_players = pd.DataFrame(r['elements'])
        df_teams = pd.DataFrame(r['teams'])[['id', 'name', 'strength']]
        
        for col in ['expected_goals', 'threat']:
            df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0)
            
        team_agg = df_players.groupby('team').agg({'expected_goals': 'mean', 'threat': 'mean'}).reset_index()
        return pd.merge(team_agg, df_teams, left_on='team', right_on='id').drop('id', axis=1)

    def fetch_fixture_map(self):
        f_data = requests.get("https://fantasy.premierleague.com/api/fixtures/?future=1").json()
        static = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        team_map = {t['id']: t['name'] for t in static['teams']}
        
        fixture_map = {}
        for f in f_data:
            h_team, a_team = team_map[f['team_h']], team_map[f['team_a']]
            fixture_map[h_team] = {"fixture_id": f['id'], "opponent": a_team}
            fixture_map[a_team] = {"fixture_id": f['id'], "opponent": h_team}
        return fixture_map

    def run_prediction_cycle(self, stats):
        print(f"[{datetime.now().strftime('%H:%M')}] Tahmin dongusu basladi...")
        f_map = self.fetch_fixture_map()
        
        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {"X-RapidAPI-Key": self.rapid_api_key, "X-RapidAPI-Host": "odds.p.rapidapi.com"}
        market_data = requests.get(url, headers=headers, params={'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}).json()

        for match in market_data:
            raw_h, raw_a = match['home_team'], match['away_team']
            norm_h, norm_a = self.normalize_name(raw_h), self.normalize_name(raw_a)
            
            f_info = f_map.get(norm_h)
            if not f_info: continue

            h_row = stats[stats['name'] == norm_h]
            a_row = stats[stats['name'] == norm_a]
            if h_row.empty or a_row.empty: continue

            h_s, a_s = h_row.iloc[0], a_row.iloc[0]
            
            # Diferansiyel Analiz ve Olasilik (Rasyonel Limitler)
            prob = np.clip(0.38 + ((h_s['strength'] - a_s['strength']) * 0.05) + ((h_s['expected_goals'] - a_s['expected_goals']) * 0.15), 0.15, 0.80)
            
            best_odds = 0
            for b in match['bookmakers']:
                for m in b['markets']:
                    if m['key'] == 'h2h':
                        for o in m['outcomes']:
                            if o['name'] == raw_h: best_odds = max(best_odds, o['price'])

            if best_odds <= 1.0: continue
            ev = (prob * best_odds) - 1

            if 0.04 < ev < 0.45:
                # Tekillik Kontrolu
                try:
                    check = requests.get(f"{self.sb_url}/rest/v1/predictions?fixture_id=eq.{f_info['fixture_id']}&team_name=eq.{norm_h}", headers=self.headers, timeout=10).json()
                    if check: 
                        print(f"⏩ {norm_h} (ID: {f_info['fixture_id']}) zaten kayitli, atlaniliyor.")
                        continue
                exceptException as e:
                    print(f"DB Kontrol Hatasi (Atlaniliyor): {e}")
                    continue

                bet = round(self.bankroll * 0.02, 2)
                payload = {
                    "fixture_id": f_info['fixture_id'], "team_name": norm_h, "model_version": "V6.2-PROD",
                    "prob_home": float(prob), "ev_value": float(ev), "placed_odds": float(best_odds),
                    "bet_amount": float(bet), "avg_xg_snapshot": float(h_s['expected_goals'])
                }
                requests.post(f"{self.sb_url}/rest/v1/predictions", headers=self.headers, json=payload)
                self.send_telegram(norm_h, norm_a, prob, best_odds, ev, bet)

    def send_telegram(self, h, a, p, o, ev, b):
        msg = (f"🛡️ *Mezalla Enterprise v6.2*\n\n🏟️ Mac: {h} vs {a}\n🎯 Tahmin: {h}\n"
               f"📊 Olasilik: %{p*100:.1f}\n💰 Oran: {o:.2f}\n📈 EV: %{ev*100:.1f}\n💵 Bahis: {b} TL")
        requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", 
                      json={"chat_id": self.tg_chat_id, "text": msg, "parse_mode": "Markdown"})

# --- MAIN EXECUTION BLOCK (MODIFIED FOR HEALTH CHECK) ---
if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M')}] Sistem tetiklendi.")
    engine = MezallaEnterprise()
    
    # --- HEALTH CHECK (HEARTBEAT) ENTEGRASYONU ---
    # Kodun calistigini teyit etmek icin Telegram'a 'canliyim' mesaji atar.
    try:
        # Markdown formatinda nice bir mesaj
        requests.post(
            f"https://api.telegram.org/bot{engine.tg_token}/sendMessage",
            json={
                "chat_id": engine.tg_chat_id,
                "text": f"🤖 *Mezalla Enterprise V6.2* Aktif.\nDenetim ve Tarama döngüsü başlatılıyor...",
                "parse_mode": "Markdown"
            },
            timeout=10 # Isletim sistemini yavaslatmamasi icin timeout ekledik
        )
        print(f"[{datetime.now().strftime('%H:%M')}] Telegram Canlilik mesaji gonderildi.")
    except Exception as e:
        print(f"Health Check Mesaj Hatasi (Sistem devam ediyor): {e}")
    # ---------------------------------------------

    # 1. Once dünün hesabını kapat.
    engine.audit_past_predictions() 
    
    # 2. Guncel veriyi cek ve temizle
    stats_data = engine.fetch_team_stats()
    
    # 3. Tahmin yap ve yeni kayitlari ac
    engine.run_prediction_cycle(stats_data) 
    
    print(f"[{datetime.now().strftime('%H:%M')}] İşlem tamamlandı.")
