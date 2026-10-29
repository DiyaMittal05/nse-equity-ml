"""
NSE Symbol Master Scraper
===========================
Downloads the complete NSE equity symbol master CSV.
"""

import requests
import csv
import os
from io import StringIO


class NSESymbolScraper:
    """Scrape NSE symbol master"""
    
    URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ohlcv')
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
    
    def fetch(self):
        """Fetch CSV text"""
        r = self.session.get(self.URL, timeout=15)
        r.raise_for_status()
        return r.text
    
    def to_csv(self, text, filename='nse_symbol_master.csv'):
        """Save raw CSV"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        
        with open(path, 'w', newline='') as f:
            f.write(text)
        
        return path
    
    def get_eq_symbols(self, text):
        """Extract EQ series symbols as a list of dicts"""
        reader = csv.DictReader(StringIO(text))
        rows = list(reader)
        eq_rows = [row for row in rows if row.get(' SERIES', '').strip() == 'EQ']
        return eq_rows
    
    def run(self):
        """Full pipeline: fetch + save + save EQ-only"""
        text = self.fetch()
        full_path = self.to_csv(text, 'nse_symbol_master.csv')
        
        eq_rows = self.get_eq_symbols(text)
        eq_path = os.path.join(self.output_dir, 'nse_symbol_master_eq.csv')
        with open(eq_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv.DictReader(StringIO(text)).fieldnames)
            writer.writeheader()
            writer.writerows(eq_rows)
        
        print(f"[NSE Scraper] Saved {len(eq_rows)} EQ symbols to {eq_path}")
        print(f"[NSE Scraper] Full file: {full_path}")
        return eq_path, eq_rows


if __name__ == '__main__':
    scraper = NSESymbolScraper()
    scraper.run()
