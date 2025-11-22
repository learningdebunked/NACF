"""Load and parse e-commerce datasets (Retailrocket, UCI)."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple


class RetailrocketLoader:
    """Load Retailrocket e-commerce dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_events(self) -> pd.DataFrame:
        """Load events.csv file."""
        try:
            events_file = self.data_path / "events.csv"
            if not events_file.exists():
                print(f"Warning: {events_file} not found. Generating synthetic data.")
                return self._generate_synthetic_data()
            
            df = pd.read_csv(events_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error loading Retailrocket data: {e}")
            return self._generate_synthetic_data()
    
    def parse_and_group(self, df: pd.DataFrame, timeout_minutes: int = 30) -> pd.DataFrame:
        """Group events by user sessions."""
        df = df.sort_values(['visitorid', 'timestamp'])
        
        # Split into sessions based on timeout
        sessions = []
        for user_id, group in df.groupby('visitorid'):
            session_id = 0
            last_time = None
            
            for idx, row in group.iterrows():
                if last_time is None or (row['timestamp'] - last_time).seconds > timeout_minutes * 60:
                    session_id += 1
                
                sessions.append({
                    'user_id': user_id,
                    'session_id': f"{user_id}_{session_id}",
                    'event_type': row['event'],
                    'item_id': row.get('itemid', -1),
                    'timestamp': row['timestamp']
                })
                last_time = row['timestamp']
        
        session_df = pd.DataFrame(sessions)
        
        # Group by session
        grouped = session_df.groupby('session_id').agg({
            'user_id': 'first',
            'event_type': lambda x: list(x),
            'timestamp': lambda x: list(x)
        }).reset_index()
        
        return grouped
    
    def _generate_synthetic_data(self, n_users: int = 1000) -> pd.DataFrame:
        """Generate synthetic e-commerce data."""
        np.random.seed(42)
        events = []
        event_types = ['view', 'addtocart', 'transaction']
        
        for user_id in range(n_users):
            n_events = np.random.randint(5, 50)
            start_time = datetime.now() - timedelta(days=np.random.randint(1, 30))
            
            for i in range(n_events):
                events.append({
                    'visitorid': user_id,
                    'timestamp': start_time + timedelta(seconds=np.random.randint(1, 300)),
                    'event': np.random.choice(event_types, p=[0.7, 0.2, 0.1]),
                    'itemid': np.random.randint(1, 1000)
                })
        
        return pd.DataFrame(events)


class UCIRetailLoader:
    """Load UCI Online Retail dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_data(self) -> pd.DataFrame:
        """Load Online_Retail.csv file."""
        try:
            retail_file = self.data_path / "Online_Retail.csv"
            if not retail_file.exists():
                print(f"Warning: {retail_file} not found. Generating synthetic data.")
                return self._generate_synthetic_data()
            
            df = pd.read_csv(retail_file, encoding='ISO-8859-1')
            df = df.dropna(subset=['CustomerID'])
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            return df
        except Exception as e:
            print(f"Error loading UCI Retail data: {e}")
            return self._generate_synthetic_data()
    
    def parse_and_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group by customer and invoice date."""
        grouped = df.groupby(['CustomerID', 'InvoiceDate']).agg({
            'StockCode': lambda x: list(x),
            'Quantity': lambda x: list(x)
        }).reset_index()
        
        grouped.columns = ['user_id', 'timestamp', 'items', 'quantities']
        return grouped
    
    def _generate_synthetic_data(self, n_customers: int = 500) -> pd.DataFrame:
        """Generate synthetic retail data."""
        np.random.seed(42)
        data = []
        
        for customer_id in range(n_customers):
            n_purchases = np.random.randint(1, 20)
            
            for _ in range(n_purchases):
                data.append({
                    'CustomerID': customer_id,
                    'InvoiceDate': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                    'StockCode': f"ITEM{np.random.randint(1, 100)}",
                    'Quantity': np.random.randint(1, 10)
                })
        
        return pd.DataFrame(data)


def session_splitter(events: List, timeout_minutes: int = 30) -> List[List]:
    """Split events into sessions based on inactivity timeout."""
    if not events:
        return []
    
    sessions = []
    current_session = [events[0]]
    
    for i in range(1, len(events)):
        time_diff = (events[i]['timestamp'] - events[i-1]['timestamp']).seconds / 60
        
        if time_diff > timeout_minutes:
            sessions.append(current_session)
            current_session = [events[i]]
        else:
            current_session.append(events[i])
    
    if current_session:
        sessions.append(current_session)
    
    return sessions


def event_encoder(event_type: str) -> int:
    """Encode event types to integers."""
    encoding = {
        'view': 0,
        'addtocart': 1,
        'transaction': 2,
        'purchase': 2,
        'click': 0,
        'cart': 1
    }
    return encoding.get(event_type.lower(), 0)
