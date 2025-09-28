"""
BusinessImpact entity model for churn prediction system.
Quantified assessment of revenue protection and cost reduction.
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class BusinessImpact:
    """Quantified assessment of revenue protection and cost reduction."""
    
    model_id: str
    predicted_churners: int
    retention_rate_improvement: float
    revenue_protection: float
    cost_reduction: float
    roi_analysis: Dict[str, float] = field(default_factory=dict)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        """Validate business impact data after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate business impact data according to business rules."""
        if not self.model_id:
            raise ValueError("model_id must be non-null and non-empty")
        
        if not isinstance(self.model_id, str):
            raise TypeError("model_id must be a string")
        
        # Validate predicted_churners is positive integer
        if not isinstance(self.predicted_churners, int) or self.predicted_churners < 0:
            raise ValueError("predicted_churners must be a non-negative integer")
        
        # Validate retention_rate_improvement is between 0 and 1
        if not isinstance(self.retention_rate_improvement, (int, float)):
            raise TypeError("retention_rate_improvement must be numeric")
        if not (0 <= self.retention_rate_improvement <= 1):
            raise ValueError(f"retention_rate_improvement must be between 0 and 1, got {self.retention_rate_improvement}")
        
        # Validate monetary values are positive
        if not isinstance(self.revenue_protection, (int, float)) or self.revenue_protection < 0:
            raise ValueError("revenue_protection must be a non-negative number")
        
        if not isinstance(self.cost_reduction, (int, float)) or self.cost_reduction < 0:
            raise ValueError("cost_reduction must be a non-negative number")
        
        # Validate ROI analysis contains valid calculations
        if not isinstance(self.roi_analysis, dict):
            raise TypeError("roi_analysis must be a dictionary")
        
        for key, value in self.roi_analysis.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"ROI analysis value for {key} must be numeric")
        
        # Validate confidence interval
        if not isinstance(self.confidence_interval, (tuple, list)) or len(self.confidence_interval) != 2:
            raise ValueError("confidence_interval must be a tuple/list of 2 values")
        
        lower, upper = self.confidence_interval
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise TypeError("confidence_interval values must be numeric")
        
        if lower > upper:
            raise ValueError(f"confidence_interval lower bound ({lower}) must be <= upper bound ({upper})")
    
    def get_total_value(self) -> float:
        """Get total business value (revenue protection + cost reduction)."""
        return self.revenue_protection + self.cost_reduction
    
    def get_value_per_customer(self) -> float:
        """Get average value per predicted churner."""
        if self.predicted_churners == 0:
            return 0.0
        return self.get_total_value() / self.predicted_churners
    
    def calculate_roi_percentage(self, investment_cost: float) -> float:
        """Calculate ROI percentage given investment cost."""
        if investment_cost <= 0:
            return float('inf') if self.get_total_value() > 0 else 0.0
        
        net_benefit = self.get_total_value() - investment_cost
        return (net_benefit / investment_cost) * 100
    
    def get_confidence_range(self) -> float:
        """Get confidence interval range (width)."""
        return self.confidence_interval[1] - self.confidence_interval[0]
    
    def is_within_confidence_bounds(self, value: float) -> bool:
        """Check if value is within confidence interval."""
        return self.confidence_interval[0] <= value <= self.confidence_interval[1]
    
    def get_revenue_protection_rate(self) -> float:
        """Get revenue protection as percentage of total value."""
        total_value = self.get_total_value()
        if total_value == 0:
            return 0.0
        return (self.revenue_protection / total_value) * 100
    
    def get_cost_reduction_rate(self) -> float:
        """Get cost reduction as percentage of total value."""
        total_value = self.get_total_value()
        if total_value == 0:
            return 0.0
        return (self.cost_reduction / total_value) * 100
    
    def update_roi_analysis(self, investment_cost: float, acquisition_cost_per_customer: float = 0) -> None:
        """Update ROI analysis with comprehensive calculations."""
        total_value = self.get_total_value()
        roi_percentage = self.calculate_roi_percentage(investment_cost)
        
        self.roi_analysis.update({
            'investment_cost': investment_cost,
            'total_value': total_value,
            'net_benefit': total_value - investment_cost,
            'roi_percentage': roi_percentage,
            'value_per_customer': self.get_value_per_customer(),
            'acquisition_cost_saved': self.predicted_churners * acquisition_cost_per_customer,
            'payback_period_months': (investment_cost / (total_value / 12)) if total_value > 0 else float('inf')
        })
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """Get executive summary for stakeholder presentation."""
        return {
            'customers_at_risk': self.predicted_churners,
            'retention_improvement': f"{self.retention_rate_improvement:.1%}",
            'total_value_protected': f"${self.get_total_value():,.2f}",
            'revenue_protection': f"${self.revenue_protection:,.2f}",
            'cost_savings': f"${self.cost_reduction:,.2f}",
            'value_per_customer': f"${self.get_value_per_customer():,.2f}",
            'confidence_range': f"±{self.get_confidence_range():.1%}",
            'roi_percentage': self.roi_analysis.get('roi_percentage', 0),
            'model_id': self.model_id
        }
    
    def is_significant_impact(self, min_value_threshold: float = 10000) -> bool:
        """Check if business impact meets significance threshold."""
        return self.get_total_value() >= min_value_threshold and self.predicted_churners > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert business impact to dictionary representation."""
        return {
            'model_id': self.model_id,
            'predicted_churners': self.predicted_churners,
            'retention_rate_improvement': self.retention_rate_improvement,
            'revenue_protection': self.revenue_protection,
            'cost_reduction': self.cost_reduction,
            'roi_analysis': self.roi_analysis,
            'confidence_interval': list(self.confidence_interval)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessImpact':
        """Create business impact from dictionary representation."""
        confidence_interval = data.get('confidence_interval', [0.0, 0.0])
        if isinstance(confidence_interval, list):
            confidence_interval = tuple(confidence_interval)
        
        return cls(
            model_id=data['model_id'],
            predicted_churners=data['predicted_churners'],
            retention_rate_improvement=data['retention_rate_improvement'],
            revenue_protection=data['revenue_protection'],
            cost_reduction=data['cost_reduction'],
            roi_analysis=data.get('roi_analysis', {}),
            confidence_interval=confidence_interval
        )
