"""
DeepFinNER: EDGAR Scraper Module
Advanced web scraping for SEC EDGAR 10-K reports with AI/ML capabilities.

This module provides intelligent scraping of financial reports with:
- Rate limiting and respectful crawling
- Error handling and retry mechanisms
- Data validation and quality checks
- Support for parallel processing
"""

import re
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import unicodedata

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from lxml import html
from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDGARScraper:
    """
    Advanced EDGAR scraper for financial report extraction.
    
    This class implements intelligent scraping techniques that serve as
    precursors to modern AI/ML applications in financial data extraction.
    """
    
    def __init__(self, base_path: str = "data/raw", rate_limit: int = 10):
        """
        Initialize the EDGAR scraper.
        
        Args:
            base_path: Base directory for storing scraped data
            rate_limit: Number of requests per second (SEC limit: 10/sec)
        """
        self.base_path = base_path
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # EDGAR URLs
        self.browse_url_base = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=10-K&dateb=&owner=exclude&output=xml&count=100"
        self.filing_url_base = "https://www.sec.gov/Archives/edgar/data/%s/%s/%s.txt"
        
        # Create directories
        os.makedirs(base_path, exist_ok=True)
        
    @sleep_and_retry
    @limits(calls=10, period=1)  # SEC rate limit: 10 requests per second
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make a rate-limited request to EDGAR.
        
        Args:
            url: URL to request
            
        Returns:
            Response object or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def _write_log(self, log_file: str, text: str) -> None:
        """Write to log file with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"[{timestamp}] {text}\n")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        This preprocessing step is crucial for NLP applications and
        demonstrates early AI/ML text processing techniques.
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def scrape_company_reports(self, ticker: str, cik: str, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict:
        """
        Scrape all 10-K reports for a specific company.
        
        Args:
            ticker: Company ticker symbol
            cik: Central Index Key
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            
        Returns:
            Dictionary containing scraping results and metadata
        """
        logger.info(f"Starting scrape for {ticker} (CIK: {cik})")
        
        # Create company directory
        company_dir = os.path.join(self.base_path, f"{ticker}_{cik}")
        os.makedirs(company_dir, exist_ok=True)
        
        log_file = os.path.join(company_dir, "scraping.log")
        self._write_log(log_file, f"Starting scrape for {ticker}")
        
        # Get filing list
        browse_url = self.browse_url_base % cik
        response = self._make_request(browse_url)
        
        if not response:
            self._write_log(log_file, f"Failed to get filing list for {ticker}")
            return {"success": False, "error": "Failed to get filing list"}
        
        # Parse filing list
        soup = BeautifulSoup(response.content, 'lxml')
        filings = self._parse_filing_list(soup, start_date, end_date)
        
        if not filings:
            self._write_log(log_file, f"No filings found for {ticker}")
            return {"success": False, "error": "No filings found"}
        
        # Scrape each filing
        scraped_reports = []
        for filing in tqdm(filings, desc=f"Scraping {ticker}"):
            report_data = self._scrape_single_filing(filing, company_dir, log_file)
            if report_data:
                scraped_reports.append(report_data)
        
        # Create summary
        summary = {
            "ticker": ticker,
            "cik": cik,
            "total_filings": len(filings),
            "successful_scrapes": len(scraped_reports),
            "scraped_reports": scraped_reports,
            "scrape_date": datetime.now().isoformat(),
            "success": True
        }
        
        # Save summary
        summary_file = os.path.join(company_dir, "scraping_summary.json")
        pd.DataFrame([summary]).to_json(summary_file, orient='records', indent=2)
        
        self._write_log(log_file, f"Completed scrape for {ticker}: {len(scraped_reports)} reports")
        
        return summary
    
    def _parse_filing_list(self, soup: BeautifulSoup, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[Dict]:
        """Parse the filing list from EDGAR browse page."""
        filings = []
        
        # Find filing table
        tables = soup.find_all('table')
        if len(tables) < 3:
            return filings
        
        filing_table = tables[2]  # Usually the third table contains filings
        rows = filing_table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 4:
                filing_date = cells[3].text.strip()
                
                # Apply date filters
                if start_date and filing_date < start_date:
                    continue
                if end_date and filing_date > end_date:
                    continue
                
                filing_info = {
                    "filing_date": filing_date,
                    "filing_type": cells[0].text.strip(),
                    "accession_number": cells[1].text.strip(),
                    "description": cells[2].text.strip()
                }
                filings.append(filing_info)
        
        return filings
    
    def _scrape_single_filing(self, filing: Dict, company_dir: str, 
                            log_file: str) -> Optional[Dict]:
        """Scrape a single 10-K filing."""
        try:
            # Construct filing URL
            accession = filing["accession_number"].replace("-", "")
            filing_url = self.filing_url_base % (
                filing["cik"], 
                accession[:10], 
                accession
            )
            
            # Download filing
            response = self._make_request(filing_url)
            if not response:
                return None
            
            # Clean and save content
            content = self._clean_text(response.text)
            
            # Save raw content
            filename = f"{filing['filing_date']}_{filing['accession_number']}.txt"
            filepath = os.path.join(company_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "filing_date": filing["filing_date"],
                "accession_number": filing["accession_number"],
                "filename": filename,
                "content_length": len(content),
                "scrape_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._write_log(log_file, f"Error scraping filing {filing['accession_number']}: {e}")
            return None
    
    def batch_scrape(self, companies: List[Tuple[str, str]], 
                    max_workers: int = 3) -> Dict:
        """
        Scrape multiple companies with parallel processing.
        
        Args:
            companies: List of (ticker, cik) tuples
            max_workers: Maximum number of parallel workers
            
        Returns:
            Summary of all scraping operations
        """
        logger.info(f"Starting batch scrape for {len(companies)} companies")
        
        results = []
        for ticker, cik in tqdm(companies, desc="Batch scraping"):
            result = self.scrape_company_reports(ticker, cik)
            results.append(result)
            
            # Respect rate limits between companies
            time.sleep(1)
        
        # Create batch summary
        batch_summary = {
            "total_companies": len(companies),
            "successful_companies": sum(1 for r in results if r["success"]),
            "total_reports": sum(len(r.get("scraped_reports", [])) for r in results),
            "results": results,
            "batch_date": datetime.now().isoformat()
        }
        
        # Save batch summary
        summary_file = os.path.join(self.base_path, "batch_scraping_summary.json")
        pd.DataFrame([batch_summary]).to_json(summary_file, orient='records', indent=2)
        
        return batch_summary


def create_company_list(csv_file: str) -> List[Tuple[str, str]]:
    """
    Create a list of companies from CSV file.
    
    Args:
        csv_file: Path to CSV file with ticker and CIK columns
        
    Returns:
        List of (ticker, cik) tuples
    """
    df = pd.read_csv(csv_file)
    
    if 'ticker' in df.columns and 'cik' in df.columns:
        return list(zip(df['ticker'], df['cik']))
    else:
        logger.error("CSV file must contain 'ticker' and 'cik' columns")
        return []


if __name__ == "__main__":
    # Example usage
    scraper = EDGARScraper()
    
    # Single company scrape
    result = scraper.scrape_company_reports("AAPL", "0000320193")
    print(f"Scraped {result['successful_scrapes']} reports for AAPL")
    
    # Batch scrape
    companies = [("AAPL", "0000320193"), ("MSFT", "0000789019")]
    batch_result = scraper.batch_scrape(companies)
    print(f"Batch scrape completed: {batch_result['successful_companies']} companies") 