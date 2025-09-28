"""
ChurnEvent entity model for churn prediction system.
Binary indicator representing customer service discontinuation.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class ChurnEvent:
    """Binary indicator representing customer service discontinuation."""
    
    customer_id: str
    churn: int  # Binary target variable (0=No Churn, 1=Churn)
    timeframe: int = 60  # Prediction window in days
    observation_date: Optional[datetime] = None
    actual_churn_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate churn event data after initialization."""
        self._validate()
        
        # Set observation_date to current time if not provided
        if self.observation_date is None:
            self.observation_date = datetime.now()
    
    def _validate(self) -> None:
        """Validate churn event data according to business rules."""
        if not self.customer_id:
            raise ValueError("customer_id must be non-null and non-empty")
        
        if not isinstance(self.customer_id, str):
            raise TypeError("customer_id must be a string")
        
        # Validate churn is binary (0 or 1)
        if self.churn not in [0, 1]:
            raise ValueError("churn must be 0 (No Churn) or 1 (Churn)")
        
        # Validate timeframe is fixed at 60 days
        if self.timeframe != 60:
            raise ValueError("timeframe must be 60 days")
        
        # Validate observation_date is datetime if provided
        if self.observation_date is not None and not isinstance(self.observation_date, datetime):
            raise TypeError("observation_date must be a datetime object")
        
        # Validate actual_churn_date logic
        if self.actual_churn_date is not None:
            if not isinstance(self.actual_churn_date, datetime):
                raise TypeError("actual_churn_date must be a datetime object")
            
            # If actual_churn_date is provided, churn should be 1
            if self.churn != 1:
                raise ValueError("actual_churn_date should only be present when churn=1")
        
        # If churn=1 and we have observation_date, actual_churn_date should be after observation_date
        if (self.churn == 1 and 
            self.observation_date is not None and 
            self.actual_churn_date is not None):
            if self.actual_churn_date <= self.observation_date:
                raise ValueError("actual_churn_date must be after observation_date")
    
    def is_churned(self) -> bool:
        """Check if customer has churned."""
        return self.churn == 1
    
    def is_within_timeframe(self, current_date: datetime) -> bool:
        """Check if current date is within the prediction timeframe."""
        if self.observation_date is None:
            return False
        
        days_elapsed = (current_date - self.observation_date).days
        return 0 <= days_elapsed <= self.timeframe
    
    def days_to_churn(self) -> Optional[int]:
        """Calculate days between observation and actual churn (if churned)."""
        if not self.is_churned() or self.actual_churn_date is None or self.observation_date is None:
            return None
        
        return (self.actual_churn_date - self.observation_date).days
    
    def time_until_prediction_window_ends(self, current_date: datetime) -> int:
        """Calculate days remaining in prediction window."""
        if self.observation_date is None:
            return 0
        
        days_elapsed = (current_date - self.observation_date).days
        return max(0, self.timeframe - days_elapsed)
    
    def to_dict(self) -> dict:
        """Convert churn event to dictionary representation."""
        return {
            'customer_id': self.customer_id,
            'churn': self.churn,
            'timeframe': self.timeframe,
            'observation_date': self.observation_date.isoformat() if self.observation_date else None,
            'actual_churn_date': self.actual_churn_date.isoformat() if self.actual_churn_date else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChurnEvent':
        """Create churn event from dictionary representation."""
        observation_date = None
        if data.get('observation_date'):
            observation_date = datetime.fromisoformat(data['observation_date'])
        
        actual_churn_date = None
        if data.get('actual_churn_date'):
            actual_churn_date = datetime.fromisoformat(data['actual_churn_date'])
        
        return cls(
            customer_id=data['customer_id'],
            churn=data['churn'],
            timeframe=data.get('timeframe', 60),
            observation_date=observation_date,
            actual_churn_date=actual_churn_date
        )
