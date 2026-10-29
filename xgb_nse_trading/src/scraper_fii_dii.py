"""
FII + DII Data Scraper
======================
Scrapes daily FII/DII cash & derivative flows from Moneycontrol.
"""

import requests
import json
import re
import csv
import os
from datetime import datetime


class MoneycontrolFIIScraper:
    """Scrape FII/DII data from Moneycontrol"""
    
    URL = "https://www.moneycontrol.com/markets/fii-dii-data/"
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'fii')
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
    
    def fetch(self):
        """Fetch raw JSON data from Moneycontrol"""
        r = self.session.get(self.URL, timeout=15)
        r.raise_for_status()
        
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', r.text, re.DOTALL)
        if not match:
            raise ValueError("Could not find __NEXT_DATA__ in Moneycontrol page")
        
        next_data = json.loads(match.group(1))
        return next_data['props']['pageProps']['FiiDiiData']['fiiDiiData']
    
    def to_csv(self, data, filename='fii_dii_historical.csv'):
        """Save scraped data to CSV"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        
        rows = []
        for item in data:
            rows.append({
                'date': item['date'],
                'fii_cm': item['fiiCM'].replace(',', ''),
                'dii_cm': item['diiCM'].replace(',', ''),
                'fii_idx_fut': item['fiiIdxFut'].replace(',', ''),
                'fii_idx_opt': item['fiiIdxOpt'].replace(',', ''),
                'fii_stk_fut': item['fiiStkFut'].replace(',', ''),
                'fii_stk_opt': item['fiiStkOpt'].replace(',', ''),
                'nifty_close': item['niftyClose'].replace(',', ''),
                'sensex_close': item['sensexClose'].replace(',', ''),
            })
        
        rows.sort(key=lambda x: x['date'])
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'fii_cm', 'dii_cm', 'fii_idx_fut', 'fii_idx_opt', 'fii_stk_fut', 'fii_stk_opt', 'nifty_close', 'sensex_close'])
            writer.writeheader()
            writer.writerows(rows)
        
        return path, rows
    
    def run(self):
        """Full pipeline: fetch + save"""
        data = self.fetch()
        path, rows = self.to_csv(data)
        print(f"[FII Scraper] Saved {len(rows)} records to {path}")
        print(f"[FII Scraper] Date range: {rows[0]['date']} to {rows[-1]['date']}")
        return path, rows


if __name__ == '__main__':
    scraper = MoneycontrolFIIScraper()
    scraper.run()
