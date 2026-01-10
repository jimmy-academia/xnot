from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum

class ScaleDomain(Enum):
    RESTAURANT = "restaurant"
    PRODUCT = "product"

@dataclass(frozen=True)
class ScaleContext:
    """Immutable context for a reasoning task."""
    id: str
    domain: ScaleDomain
    metadata: Dict[str, Any]  # Attributes (price, wifi, etc.)
    reviews: List[Dict[str, Any]]  # Standardized review list
    
    @property
    def text_blob(self) -> str:
        """Lazy concatenation of reviews for 'string-mode' LLMs."""
        return "\\n".join([f"Review {i+1}: {r['text']}" for i, r in enumerate(self.reviews)])

@dataclass
class ScaleResult:
    """Standardized result object for all SCALE tasks."""
    task_id: str
    verdict: int  # 0 or 1
    verdict_prediction: Any
    verdict_ground_truth: Any
    evidence_score: float  # 0.0 to 1.0 (mean of premise scores)
    premises: Dict[str, Dict[str, Any]]  # {premise_key: {pred, gt, score}}
    
    @property
    def final_score(self) -> float:
        """The core SCALE metric: Verdict * Evidence."""
        return self.verdict * self.evidence_score

class ScaleTask(ABC):
    """Abstract Base Class for all 100 SCALE tasks."""
    
    def __init__(self, task_id: str, query_template: str):
        self.task_id = task_id
        self.query_template = query_template
        
    @abstractmethod
    def get_query(self, context: ScaleContext) -> str:
        """Format the natural language query for this context."""
        pass
        
    @abstractmethod
    def compute_ground_truth(self, context: ScaleContext) -> Dict[str, Any]:
        """
        Return dict with:
        - 'verdict': 0/1 or True/False
        - 'premises': {key: value}
        """
        pass
        
    def score(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> ScaleResult:
        """Compare prediction to ground truth and return ScaleResult."""
        # This will vary slightly by task type (Exact Match vs Float Tolerance)
        pass
