import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

class MezallaConsistentEngine:
    def __init__(self):
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
        # Risk Yonetimi Parametreleri
        self.max_odds = 3.00       # Gökdelen oranlari engelle
        self.min_true_prob = 0.35  # %35 kazanma ihtimali altini alma
        self.min_ev = 0.02         # %2 ve uzeri avantaj yeterli

    def fetch_market_odds(self):
        print(f"[{datetime.now().strftime('%H:%M')}] Piyasa H2H verileri toplaniyor...")
        if not self.rapid_api_key: return []

        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": "odds.p.rapidapi.com"
        }
        params = {'regions': 'eu,uk,us', 'markets': 'h2h', 'oddsFormat': 'decimal'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            return response.json() if response.status_code == 200 else []
        except: return []

    def analyze_value(self, data):
        print(f"[{datetime.now().strftime('%H:%M')}] Yuksek isabetli sinyal analizi yapiliyor...")
        opportunities = []

        for match in data:
            match_name = f"{match['home_team']} vs {match['away_team']}"
            odds_pool = {}
            
            for bookie in match.get('bookmakers', []):
                for market in bookie.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            name = outcome['name']
                            price = outcome['price']
                            if name not in odds_pool:
                                odds_pool[name] = {'prices': [], 'best_price': 0, 'best_bookie': ''}
                            odds_pool[name]['prices'].append(price)
                            if price > odds_pool[name]['best_price']:
                                odds_pool[name]['best_price'] = price
                                odds_pool[name]['best_bookie'] = bookie['title']

            for name, stats in odds_pool.items():
                if len(stats['prices']) < 3: continue
                
                avg_odds = np.mean(stats['prices'])
                true_prob = 1 / avg_odds
                ev = (true_prob * stats['best_price']) - 1
                
                # --- KRITIK FILTRELER ---
                # 1. Oran cok yuksek olmamali (Basari orani icin)
                # 2. Gercek kazanc ihtimali yuksek olmali
                # 3. Hala matematiksel avantaj (EV) olmali
                if stats['best_price'] <= self.max_odds and true_prob >= self.min_true_prob and ev > self.min_ev:
                    opportunities.append({
                        'match': match_name,
                        'selection': name,
                        'prob': true_prob,
                        'odds': stats['best_price'],
                        'bookie': stats['best_bookie'],
                        'ev': ev
                    })

        return pd.DataFrame(opportunities)

    def send_notification(self, message):
        if not self.tg_token or not self.tg_chat_id: return
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        requests.post(url, json={"chat_id": self.tg_chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)

    def run_cycle(self):
        raw_data = self.fetch_market_odds()
        if not raw_data: return

        signals = self.analyze_value(raw_data)
        
        if not signals.empty:
            # En guvenilir (en yuksek olasilikli) 3 sinyali sec
            signals = signals.sort_values(by='prob', ascending=False).head(3)
            
            msg = "🎯 *Mezalla Konsantre Rapor (Yuksek Basari)*\n\n"
            for _, row in signals.iterrows():
                # Olasilik yuksekse bahis miktarini hafif artirabiliriz (Kelly benzeri mantik)
                bet = np.round(self.bankroll * 0.03, 2) 
                msg += (f"⚽ *{row['match']}*\n"
                        f"✅ Tahmin: {row['selection']}\n"
                        f"📊 Olasilik: %{row['prob']*100:.1f}\n"
                        f"💰 Oran: {row['odds']:.2f} ({row['bookie']})\n"
                        f"💵 Bahis: {bet} TL\n\n")
            
            self.send_notification(msg)
        else:
            print("Guvenli ve degerli firsat bulunamadi.")

if __name__ == "__main__":
    engine = MezallaConsistentEngine()
    engine.run_cycle()
