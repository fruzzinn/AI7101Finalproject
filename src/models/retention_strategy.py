"""
RetentionStrategy entity model for churn prediction system.
Actionable recommendations derived from model insights.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class StrategyType(Enum):
    """Types of retention strategies."""
    DISCOUNT = "discount"
    UPGRADE = "upgrade"
    ENGAGEMENT = "engagement"
    SUPPORT = "support"
    PERSONALIZATION = "personalization"
    LOYALTY = "loyalty"


class UrgencyLevel(Enum):
    """Urgency levels for strategy implementation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RetentionStrategy:
    """Actionable recommendations derived from model insights."""
    
    strategy_id: str
    target_segment: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    priority_score: float = 0.0
    expected_success_rate: float = 0.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    timeline: Dict[str, Any] = field(default_factory=dict)
    strategy_type: Optional[StrategyType] = None
    urgency_level: Optional[UrgencyLevel] = None
    
    def __post_init__(self):
        """Validate retention strategy data after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate retention strategy data according to business rules."""
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-null and non-empty")
        
        if not isinstance(self.strategy_id, str):
            raise TypeError("strategy_id must be a string")
        
        # Validate priority score is between 0 and 1
        if not isinstance(self.priority_score, (int, float)):
            raise TypeError("priority_score must be numeric")
        if not (0 <= self.priority_score <= 1):
            raise ValueError(f"priority_score must be between 0 and 1, got {self.priority_score}")
        
        # Validate expected success rate is between 0 and 1
        if not isinstance(self.expected_success_rate, (int, float)):
            raise TypeError("expected_success_rate must be numeric")
        if not (0 <= self.expected_success_rate <= 1):
            raise ValueError(f"expected_success_rate must be between 0 and 1, got {self.expected_success_rate}")
        
        # Validate target_segment is a dictionary
        if not isinstance(self.target_segment, dict):
            raise TypeError("target_segment must be a dictionary")
        
        # Validate recommended_actions is a list of strings
        if not isinstance(self.recommended_actions, list):
            raise TypeError("recommended_actions must be a list")
        
        for i, action in enumerate(self.recommended_actions):
            if not isinstance(action, str):
                raise TypeError(f"recommended_actions[{i}] must be a string")
        
        # Validate resource_requirements contains cost estimates
        if not isinstance(self.resource_requirements, dict):
            raise TypeError("resource_requirements must be a dictionary")
        
        # If resource requirements are provided, validate they include cost
        if self.resource_requirements and 'cost_estimate' not in self.resource_requirements:
            raise ValueError("resource_requirements must include cost_estimate")
        
        # Validate timeline includes start and end dates
        if not isinstance(self.timeline, dict):
            raise TypeError("timeline must be a dictionary")
        
        if self.timeline:
            required_timeline_keys = ['start_date', 'end_date']
            for key in required_timeline_keys:
                if key not in self.timeline:
                    raise ValueError(f"timeline must include {key}")
            
            # Validate date objects if provided
            for date_key in ['start_date', 'end_date']:
                if date_key in self.timeline:
                    date_value = self.timeline[date_key]
                    if not isinstance(date_value, (datetime, date, str)):
                        raise TypeError(f"timeline {date_key} must be a date, datetime, or ISO string")
    
    def get_urgency_from_priority(self) -> UrgencyLevel:
        """Determine urgency level based on priority score."""
        if self.priority_score >= 0.8:
            return UrgencyLevel.CRITICAL
        elif self.priority_score >= 0.6:
            return UrgencyLevel.HIGH
        elif self.priority_score >= 0.4:
            return UrgencyLevel.MEDIUM
        else:
            return UrgencyLevel.LOW
    
    def estimate_customer_impact(self, total_customers: int) -> int:
        """Estimate number of customers this strategy could impact."""
        if not self.target_segment:
            return 0
        
        # Simple estimation based on segment size (default 10% if not specified)
        segment_percentage = self.target_segment.get('percentage', 0.1)
        return int(total_customers * segment_percentage)
    
    def calculate_expected_retention(self, at_risk_customers: int) -> int:
        """Calculate expected number of customers retained."""
        impact_customers = min(at_risk_customers, self.estimate_customer_impact(at_risk_customers))
        return int(impact_customers * self.expected_success_rate)
    
    def get_cost_per_customer(self) -> float:
        """Get estimated cost per customer for this strategy."""
        if not self.resource_requirements or 'cost_estimate' not in self.resource_requirements:
            return 0.0
        
        total_cost = self.resource_requirements['cost_estimate']
        customers_impacted = self.target_segment.get('customer_count', 1)
        
        return total_cost / max(customers_impacted, 1)
    
    def get_roi_estimate(self, average_customer_value: float) -> float:
        """Calculate estimated ROI for this strategy."""
        cost_per_customer = self.get_cost_per_customer()
        if cost_per_customer == 0:
            return float('inf')
        
        value_retained = average_customer_value * self.expected_success_rate
        return (value_retained - cost_per_customer) / cost_per_customer
    
    def add_action(self, action: str) -> None:
        """Add a recommended action to the strategy."""
        if not isinstance(action, str):
            raise TypeError("Action must be a string")
        if action not in self.recommended_actions:
            self.recommended_actions.append(action)
    
    def set_target_segment(self, segment_criteria: Dict[str, Any]) -> None:
        """Set target customer segment criteria."""
        if not isinstance(segment_criteria, dict):
            raise TypeError("Segment criteria must be a dictionary")
        self.target_segment.update(segment_criteria)
    
    def set_timeline(self, start_date: Any, end_date: Any) -> None:
        """Set implementation timeline."""
        self.timeline['start_date'] = start_date
        self.timeline['end_date'] = end_date
        self._validate()  # Re-validate after setting timeline
    
    def set_resource_requirements(self, cost_estimate: float, **other_requirements) -> None:
        """Set resource requirements including cost estimate."""
        if not isinstance(cost_estimate, (int, float)) or cost_estimate < 0:
            raise ValueError("cost_estimate must be a non-negative number")
        
        self.resource_requirements['cost_estimate'] = cost_estimate
        self.resource_requirements.update(other_requirements)
    
    def is_high_priority(self, threshold: float = 0.7) -> bool:
        """Check if strategy is high priority."""
        return self.priority_score >= threshold
    
    def is_cost_effective(self, max_cost_per_customer: float) -> bool:
        """Check if strategy is cost-effective."""
        return self.get_cost_per_customer() <= max_cost_per_customer
    
    def get_implementation_summary(self) -> Dict[str, Any]:
        """Get summary for implementation planning."""
        urgency = self.urgency_level or self.get_urgency_from_priority()
        
        return {
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type.value if self.strategy_type else None,
            'urgency_level': urgency.value,
            'priority_score': self.priority_score,
            'expected_success_rate': f"{self.expected_success_rate:.1%}",
            'actions_count': len(self.recommended_actions),
            'cost_per_customer': f"${self.get_cost_per_customer():.2f}",
            'target_segment_size': self.target_segment.get('customer_count', 'Unknown'),
            'timeline_duration': self._get_timeline_duration(),
            'is_high_priority': self.is_high_priority()
        }
    
    def _get_timeline_duration(self) -> str:
        """Calculate timeline duration in days."""
        if 'start_date' not in self.timeline or 'end_date' not in self.timeline:
            return "Unknown"
        
        try:
            start = self.timeline['start_date']
            end = self.timeline['end_date']
            
            # Convert to datetime objects if they're strings
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            if isinstance(end, str):
                end = datetime.fromisoformat(end)
            
            # Convert date to datetime if needed
            if isinstance(start, date) and not isinstance(start, datetime):
                start = datetime.combine(start, datetime.min.time())
            if isinstance(end, date) and not isinstance(end, datetime):
                end = datetime.combine(end, datetime.min.time())
            
            duration = (end - start).days
            return f"{duration} days"
        except (ValueError, TypeError):
            return "Invalid dates"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert retention strategy to dictionary representation."""
        return {
            'strategy_id': self.strategy_id,
            'target_segment': self.target_segment,
            'recommended_actions': self.recommended_actions,
            'priority_score': self.priority_score,
            'expected_success_rate': self.expected_success_rate,
            'resource_requirements': self.resource_requirements,
            'timeline': self.timeline,
            'strategy_type': self.strategy_type.value if self.strategy_type else None,
            'urgency_level': self.urgency_level.value if self.urgency_level else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetentionStrategy':
        """Create retention strategy from dictionary representation."""
        strategy_type = None
        if data.get('strategy_type'):
            strategy_type = StrategyType(data['strategy_type'])
        
        urgency_level = None
        if data.get('urgency_level'):
            urgency_level = UrgencyLevel(data['urgency_level'])
        
        return cls(
            strategy_id=data['strategy_id'],
            target_segment=data.get('target_segment', {}),
            recommended_actions=data.get('recommended_actions', []),
            priority_score=data.get('priority_score', 0.0),
            expected_success_rate=data.get('expected_success_rate', 0.0),
            resource_requirements=data.get('resource_requirements', {}),
            timeline=data.get('timeline', {}),
            strategy_type=strategy_type,
            urgency_level=urgency_level
        )
