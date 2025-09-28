"""
Customer entity model for churn prediction system.
Represents individual telecommunications subscriber with associated features.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Customer:
    """Individual telecommunications subscriber with associated features."""
    
    customer_id: str
    demographic_features: Dict[str, Any] = field(default_factory=dict)
    usage_patterns: Dict[str, Any] = field(default_factory=dict)
    service_history: Dict[str, Any] = field(default_factory=dict)
    billing_information: Dict[str, Any] = field(default_factory=dict)
    support_interactions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate customer data after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate customer data according to business rules."""
        if not self.customer_id:
            raise ValueError("customer_id must be non-null and non-empty")
        
        if not isinstance(self.customer_id, str):
            raise TypeError("customer_id must be a string")
        
        # Validate feature dictionaries contain valid values
        for feature_name, feature_dict in [
            ('demographic_features', self.demographic_features),
            ('usage_patterns', self.usage_patterns),
            ('service_history', self.service_history),
            ('billing_information', self.billing_information),
            ('support_interactions', self.support_interactions)
        ]:
            if not isinstance(feature_dict, dict):
                raise TypeError(f"{feature_name} must be a dictionary")
    
    def add_demographic_feature(self, key: str, value: Any) -> None:
        """Add demographic feature (age, gender, location, account_type)."""
        self.demographic_features[key] = value
    
    def add_usage_pattern(self, key: str, value: Any) -> None:
        """Add usage pattern (call frequency, data usage, SMS volume)."""
        self.usage_patterns[key] = value
    
    def add_service_history(self, key: str, value: Any) -> None:
        """Add service history (tenure, service changes, upgrades)."""
        self.service_history[key] = value
    
    def add_billing_info(self, key: str, value: Any) -> None:
        """Add billing information (payment method, bill amount, payment history)."""
        self.billing_information[key] = value
    
    def add_support_interaction(self, key: str, value: Any) -> None:
        """Add support interaction (call center contacts, complaint history)."""
        self.support_interactions[key] = value
    
    def get_all_features(self) -> Dict[str, Any]:
        """Get all features as a flattened dictionary."""
        all_features = {}
        all_features.update(self.demographic_features)
        all_features.update(self.usage_patterns)
        all_features.update(self.service_history)
        all_features.update(self.billing_information)
        all_features.update(self.support_interactions)
        return all_features
    
    def has_missing_values(self) -> bool:
        """Check if customer has any missing feature values."""
        all_features = self.get_all_features()
        return any(value is None or value == '' for value in all_features.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert customer to dictionary representation."""
        return {
            'customer_id': self.customer_id,
            'demographic_features': self.demographic_features,
            'usage_patterns': self.usage_patterns,
            'service_history': self.service_history,
            'billing_information': self.billing_information,
            'support_interactions': self.support_interactions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Customer':
        """Create customer from dictionary representation."""
        return cls(
            customer_id=data['customer_id'],
            demographic_features=data.get('demographic_features', {}),
            usage_patterns=data.get('usage_patterns', {}),
            service_history=data.get('service_history', {}),
            billing_information=data.get('billing_information', {}),
            support_interactions=data.get('support_interactions', {})
        )
