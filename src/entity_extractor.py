"""
DeepFinNER: Entity Extractor Module
Advanced Named Entity Recognition for financial documents.

This module implements sophisticated NER techniques that serve as
precursors to modern AI/ML applications in financial entity extraction.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialEntityExtractor:
    """
    Advanced Named Entity Recognition for financial documents.
    
    This class implements sophisticated NER techniques that demonstrate
    early AI/ML capabilities in financial entity extraction.
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the financial entity extractor.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NER
        """
        self.use_spacy = use_spacy
        
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        
        # Financial entity patterns
        self.financial_patterns = {
            'currency_amount': [
                r'\$[\d,]+(?:\.\d{2})?',
                r'[\d,]+(?:\.\d{2})?\s*(?:dollars?|usd)',
                r'(?:dollars?|usd)\s*[\d,]+(?:\.\d{2})?'
            ],
            'percentage': [
                r'\d+(?:\.\d+)?%',
                r'\d+(?:\.\d+)?\s*percent',
                r'percent\s*\d+(?:\.\d+)?'
            ],
            'date': [
                r'\d{1,2}/\d{1,2}/\d{2,4}',
                r'\d{4}-\d{2}-\d{2}',
                r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
            ],
            'fiscal_year': [
                r'fiscal\s+year\s+\d{4}',
                r'fy\s+\d{4}',
                r'\d{4}\s+fiscal\s+year'
            ],
            'quarter': [
                r'q[1-4]\s+\d{4}',
                r'\d{4}\s+q[1-4]',
                r'(?:first|second|third|fourth)\s+quarter\s+\d{4}'
            ],
            'financial_metrics': [
                r'\b(?:revenue|sales|income|profit|loss|earnings|ebitda|ebit|gross\s+profit|net\s+income)\b',
                r'\b(?:assets|liabilities|equity|debt|cash|inventory|receivables|payables)\b',
                r'\b(?:roi|roe|roa|eps|pe\s+ratio|debt\s+to\s+equity)\b'
            ],
            'company_names': [
                r'\b(?:inc|corp|corporation|company|ltd|llc|limited)\b',
                r'\b(?:apple|microsoft|google|amazon|facebook|tesla|netflix)\b'
            ],
            'market_terms': [
                r'\b(?:stock|share|dividend|market\s+cap|valuation|ipo|merger|acquisition)\b',
                r'\b(?:bull\s+market|bear\s+market|volatility|beta|alpha|sharpe\s+ratio)\b'
            ]
        }
        
        # Financial entity dictionaries
        self.financial_entities = {
            'financial_ratios': {
                'profitability': ['roi', 'roe', 'roa', 'gross_margin', 'net_margin', 'operating_margin'],
                'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio'],
                'solvency': ['debt_to_equity', 'debt_to_assets', 'interest_coverage'],
                'efficiency': ['asset_turnover', 'inventory_turnover', 'receivables_turnover']
            },
            'financial_statements': {
                'income_statement': ['revenue', 'cost_of_goods_sold', 'gross_profit', 'operating_expenses', 'net_income'],
                'balance_sheet': ['assets', 'liabilities', 'equity', 'cash', 'inventory', 'receivables'],
                'cash_flow': ['operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow']
            },
            'market_indicators': {
                'valuation': ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda'],
                'growth': ['revenue_growth', 'earnings_growth', 'market_share'],
                'risk': ['beta', 'volatility', 'var', 'sharpe_ratio']
            }
        }
    
    def extract_entities_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities using pattern matching.
        
        This demonstrates early NER techniques that paved the way
        for modern AI/ML applications in entity extraction.
        """
        entities = {}
        
        for entity_type, patterns in self.financial_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                # Clean and deduplicate matches
                cleaned_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        cleaned_matches.extend([m.strip() for m in match if m.strip()])
                    else:
                        cleaned_matches.append(match.strip())
                
                entities[entity_type] = list(set(cleaned_matches))
        
        return entities
    
    def extract_entities_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using spaCy's advanced NER.
        
        This demonstrates modern NER techniques that are fundamental
        to current AI/ML applications.
        """
        if not self.use_spacy:
            return {}
        
        doc = self.nlp(text)
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entity_type = ent.label_.lower()
            entity_text = ent.text.strip()
            
            # Filter for financial relevance
            if self._is_financially_relevant(entity_type, entity_text):
                entities[entity_type].append(entity_text)
        
        # Deduplicate
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return dict(entities)
    
    def _is_financially_relevant(self, entity_type: str, entity_text: str) -> bool:
        """Check if an entity is financially relevant."""
        financial_keywords = {
            'org': ['inc', 'corp', 'ltd', 'company', 'bank', 'financial', 'investment'],
            'money': ['dollar', 'euro', 'pound', 'yen', 'million', 'billion'],
            'percent': ['percent', 'percentage', 'rate', 'ratio'],
            'date': ['fiscal', 'quarter', 'annual', 'year'],
            'cardinal': ['million', 'billion', 'thousand', 'hundred']
        }
        
        if entity_type in financial_keywords:
            return any(keyword in entity_text.lower() for keyword in financial_keywords[entity_type])
        
        return True
    
    def extract_financial_metrics(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract structured financial metrics with context.
        
        This demonstrates advanced entity extraction techniques that
        are crucial for modern AI/ML applications in finance.
        """
        metrics = {
            'revenue_metrics': [],
            'profitability_metrics': [],
            'balance_sheet_metrics': [],
            'market_metrics': []
        }
        
        # Revenue patterns
        revenue_patterns = [
            r'(?:revenue|sales)\s+(?:of\s+)?(\$[\d,]+(?:\.\d{2})?)\s+(?:million|billion)?',
            r'(\$[\d,]+(?:\.\d{2})?)\s+(?:million|billion)?\s+(?:in\s+)?(?:revenue|sales)',
            r'(?:revenue|sales)\s+(?:increased|decreased|grew)\s+by\s+(\d+(?:\.\d+)?%)'
        ]
        
        for pattern in revenue_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                metrics['revenue_metrics'].append({
                    'value': match.group(1),
                    'context': match.group(0),
                    'position': match.span()
                })
        
        # Profitability patterns
        profit_patterns = [
            r'(?:net\s+income|profit|earnings)\s+(?:of\s+)?(\$[\d,]+(?:\.\d{2})?)\s+(?:million|billion)?',
            r'(\$[\d,]+(?:\.\d{2})?)\s+(?:million|billion)?\s+(?:in\s+)?(?:net\s+income|profit|earnings)',
            r'(?:profit\s+margin|net\s+margin)\s+(?:of\s+)?(\d+(?:\.\d+)?%)'
        ]
        
        for pattern in profit_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                metrics['profitability_metrics'].append({
                    'value': match.group(1),
                    'context': match.group(0),
                    'position': match.span()
                })
        
        # Balance sheet patterns
        balance_patterns = [
            r'(?:total\s+assets|assets)\s+(?:of\s+)?(\$[\d,]+(?:\.\d{2})?)\s+(?:million|billion)?',
            r'(?:total\s+liabilities|debt)\s+(?:of\s+)?(\$[\d,]+(?:\.\d{2})?)\s+(?:million|billion)?',
            r'(?:cash\s+and\s+cash\s+equivalents|cash)\s+(?:of\s+)?(\$[\d,]+(?:\.\d{2})?)\s+(?:million|billion)?'
        ]
        
        for pattern in balance_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                metrics['balance_sheet_metrics'].append({
                    'value': match.group(1),
                    'context': match.group(0),
                    'position': match.span()
                })
        
        # Market patterns
        market_patterns = [
            r'(?:market\s+cap|market\s+capitalization)\s+(?:of\s+)?(\$[\d,]+(?:\.\d{2})?)\s+(?:million|billion)?',
            r'(?:pe\s+ratio|price\s+to\s+earnings)\s+(?:of\s+)?(\d+(?:\.\d+)?)',
            r'(?:stock\s+price|share\s+price)\s+(?:of\s+)?(\$[\d,]+(?:\.\d{2})?)'
        ]
        
        for pattern in market_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                metrics['market_metrics'].append({
                    'value': match.group(1),
                    'context': match.group(0),
                    'position': match.span()
                })
        
        return metrics
    
    def extract_risk_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract risk-related entities from financial text.
        
        This demonstrates specialized entity extraction for risk analysis,
        a crucial component of modern AI/ML applications in finance.
        """
        risk_patterns = {
            'risk_factors': [
                r'\b(?:risk|risks|uncertainty|uncertainties)\b',
                r'\b(?:volatility|fluctuation|variability)\b',
                r'\b(?:exposure|vulnerability|susceptibility)\b'
            ],
            'market_risks': [
                r'\b(?:market\s+risk|systematic\s+risk|beta\s+risk)\b',
                r'\b(?:interest\s+rate\s+risk|currency\s+risk|commodity\s+risk)\b',
                r'\b(?:liquidity\s+risk|credit\s+risk|default\s+risk)\b'
            ],
            'operational_risks': [
                r'\b(?:operational\s+risk|business\s+risk|strategic\s+risk)\b',
                r'\b(?:regulatory\s+risk|compliance\s+risk|legal\s+risk)\b',
                r'\b(?:technology\s+risk|cyber\s+risk|data\s+risk)\b'
            ],
            'financial_risks': [
                r'\b(?:financial\s+risk|leverage\s+risk|debt\s+risk)\b',
                r'\b(?:cash\s+flow\s+risk|funding\s+risk|refinancing\s+risk)\b',
                r'\b(?:concentration\s+risk|diversification\s+risk)\b'
            ]
        }
        
        risk_entities = {}
        
        for risk_type, patterns in risk_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                risk_entities[risk_type] = list(set(matches))
        
        return risk_entities
    
    def extract_temporal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract temporal entities for time-series analysis.
        
        This demonstrates temporal entity extraction techniques that
        are essential for modern AI/ML applications in financial forecasting.
        """
        temporal_patterns = {
            'dates': [
                r'\d{1,2}/\d{1,2}/\d{2,4}',
                r'\d{4}-\d{2}-\d{2}',
                r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
            ],
            'quarters': [
                r'q[1-4]\s+\d{4}',
                r'\d{4}\s+q[1-4]',
                r'(?:first|second|third|fourth)\s+quarter\s+\d{4}'
            ],
            'fiscal_periods': [
                r'fiscal\s+year\s+\d{4}',
                r'fy\s+\d{4}',
                r'\d{4}\s+fiscal\s+year'
            ],
            'relative_time': [
                r'(?:previous|prior|last)\s+(?:year|quarter|month)',
                r'(?:next|upcoming|following)\s+(?:year|quarter|month)',
                r'(?:current|present)\s+(?:year|quarter|month)'
            ]
        }
        
        temporal_entities = {}
        
        for time_type, patterns in temporal_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                temporal_entities[time_type] = list(set(matches))
        
        return temporal_entities
    
    def comprehensive_entity_extraction(self, text: str) -> Dict:
        """
        Perform comprehensive entity extraction on financial text.
        
        This method demonstrates a complete NER pipeline that showcases
        early AI/ML capabilities in financial entity extraction.
        """
        results = {
            'pattern_entities': self.extract_entities_with_patterns(text),
            'spacy_entities': self.extract_entities_with_spacy(text),
            'financial_metrics': self.extract_financial_metrics(text),
            'risk_entities': self.extract_risk_entities(text),
            'temporal_entities': self.extract_temporal_entities(text),
            'extraction_metadata': {
                'text_length': len(text),
                'extraction_timestamp': datetime.now().isoformat(),
                'methods_used': ['pattern_matching', 'spacy_ner'] if self.use_spacy else ['pattern_matching']
            }
        }
        
        # Calculate entity density
        total_entities = sum(len(entities) for entities in results['pattern_entities'].values())
        results['entity_density'] = total_entities / len(text.split()) if text.split() else 0
        
        return results


def batch_extract_entities(documents: List[Dict], extractor: FinancialEntityExtractor) -> List[Dict]:
    """
    Extract entities from multiple documents in batch.
    
    Args:
        documents: List of document dictionaries with 'text' key
        extractor: FinancialEntityExtractor instance
        
    Returns:
        List of entity extraction results
    """
    results = []
    
    for doc in documents:
        try:
            extracted = extractor.comprehensive_entity_extraction(doc['text'])
            extracted['document_id'] = doc.get('id', 'unknown')
            extracted['document_date'] = doc.get('date', 'unknown')
            results.append(extracted)
        except Exception as e:
            logger.error(f"Error extracting entities from document {doc.get('id', 'unknown')}: {e}")
            continue
    
    return results


if __name__ == "__main__":
    # Example usage
    extractor = FinancialEntityExtractor()
    
    # Sample financial text
    sample_text = """
    Apple Inc. reported revenue of $394.3 billion for fiscal year 2022, 
    representing an 8% increase compared to the previous year. The company's 
    net income reached $99.8 billion, driven by strong iPhone sales and 
    services growth. However, supply chain challenges and macroeconomic 
    uncertainty pose risks to future performance. The stock price closed at 
    $150.23 per share, with a PE ratio of 25.4.
    """
    
    # Extract entities
    entities = extractor.comprehensive_entity_extraction(sample_text)
    
    print("Pattern Entities:", entities['pattern_entities'])
    print("Financial Metrics:", entities['financial_metrics'])
    print("Risk Entities:", entities['risk_entities'])
    print("Entity Density:", entities['entity_density']) 