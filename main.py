import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

class MezallaH2HEngine:
    def __init__(self):
        # Cevresel Degiskenler (GitHub Secrets)
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
        self.target_market = 'h2h' # Ucretsiz Planda Acik Olan Market

    def fetch_market_odds(self):
        """RapidAPI uzerinden tum burolarin H2H oranlarini ceker."""
        print(f"[{datetime.now().strftime('%H:%M')}] Piyasa H2H oranlari cekiliyor...")
        
        if not self.rapid_api_key:
            print("HATA: ODDS_API_KEY bulunamadi.")
            return []

        url = "https://odds.p.rapidapi.com/v4/sports/soccer_epl/odds"
        
        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": "odds.p.rapidapi.com"
        }
        
        params = {
            'regions': 'eu,us,uk',
            'markets': self.target_market,
            'oddsFormat': 'decimal'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code != 200:
                print(f"API HATASI ({response.status_code}): {response.text}")
                return []
                
            return response.json()
        except Exception as e:
            print(f"Baglanti Hatasi: {e}")
            return []

    def calculate_value_bets(self, data):
        """Burolar arasi oran farklarini kullanarak EV (Expected Value) hesaplar."""
        print(f"[{datetime.now().strftime('%H:%M')}] Oran analizi ve EV hesaplamasi yapiliyor...")
        opportunities = []

        for match in data:
            match_name = f"{match['home_team']} vs {match['away_team']}"
            kickoff = match['commence_time']
            
            # Her ciktinin (Home, Away, Draw) oranlarini topla
            odds_pool = {}
            for bookie in match.get('bookmakers', []):
                for market in bookie.get('markets', []):
                    if market['key'] == self.target_market:
                        for outcome in market['outcomes']:
                            name = outcome['name']
                            price = outcome['price']
                            if name not in odds_pool:
                                odds_pool[name] = {'prices': [], 'best_price': 0, 'best_bookie': ''}
                            
                            odds_pool[name]['prices'].append(price)
                            
                            if price > odds_pool[name]['best_price']:
                                odds_pool[name]['best_price'] = price
                                odds_pool[name]['best_bookie'] = bookie['title']

            # Beklenen Deger (EV) Hesaplamasi
            for name, stats in odds_pool.items():
                if len(stats['prices']) < 3:
                    continue # Yeterli piyasa verisi yoksa atla
                
                # Ortalama oran piyasanin "gercek" beklentisini yansitir
                avg_odds = np.mean(stats['prices'])
                true_prob = 1 / avg_odds
                
                # En yuksek orani gercek olasilikla carp
                ev = (true_prob * stats['best_price']) - 1
                
                # %3 (0.03) uzeri kâr marji olanlari (Değer Bahisleri) filtrele
                if ev > 0.03:
                    opportunities.append({
                        'match': match_name,
                        'selection': name,
                        'true_prob': true_prob,
                        'best_odds': stats['best_price'],
                        'bookie': stats['best_bookie'],
                        'ev': ev,
                        'kickoff': kickoff
                    })

        return pd.DataFrame(opportunities)

    def send_notification(self, message):
        """Telegram uzerinden otonom rapor iletir."""
        if not self.tg_token or not self.tg_chat_id: 
            print("Telegram API bilgileri eksik, mesaj gonderilemedi.")
            return
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.tg_chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print(f"Telegram Hatasi: {e}")

    def run_forecast_cycle(self):
        raw_data = self.fetch_market_odds()
        if not raw_data:
            print("Piyasa verisi alinamadi. Otonom dongu durduruldu.")
            return

        df_signals = self.calculate_value_bets(raw_data)
        
        if not df_signals.empty:
            # En yuksek EV'ye sahip firsatlari sirala
            df_signals = df_signals.sort_values(by='ev', ascending=False).head(5)
            
            msg = f"🚀 *Mezalla H2H Arbitraj Raporu*\n\n"
            
            for _, row in df_signals.iterrows():
                bet_amount = np.round(np.minimum(self.bankroll * 0.02, 25.0), 2) # %2 kasa yonetimi
                msg += (f"⚽ *{row['match']}*\n"
                        f"🎯 Tahmin: *{row['selection']}*\n"
                        f"🏦 Buro: {row['bookie']}\n"
                        f"💰 Oran: {row['best_odds']:.2f}\n"
                        f"📈 EV (Değer): %{row['ev']*100:.1f}\n"
                        f"💵 Onerilen Bahis: {bet_amount} TL\n\n")
            
            self.send_notification(msg)
            print("Degerli firsatlar bulundu ve Telegram'a iletildi.")
        else:
            print("Kriterlere uygun, kâr marji (EV) %3'ten yuksek H2H bahsi bulunamadi.")

if __name__ == "__main__":
    engine = MezallaH2HEngine()
    engine.run_forecast_cycle()
