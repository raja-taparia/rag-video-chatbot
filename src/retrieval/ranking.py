# Ranking strategies for retrieval results
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class RankingStrategy:
    """Base interface for ranking retrieval results."""
    def rank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement rank method")

class CosineSimilarityRanking(RankingStrategy):
    """Default ranking using cosine similarity scores."""
    def __init__(self, min_score: float = 0.0, normalize_scores: bool = True):
        self.min_score = min_score
        self.normalize_scores = normalize_scores
        logger.info(f"CosineSimilarityRanking initialized (min_score={min_score})")
    
    def rank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Filter, sort, normalize scores
        filtered = [r for r in results if r.get('score', 0.0) >= self.min_score]
        sorted_results = sorted(filtered, key=lambda x: x.get('score', 0.0), reverse=True)
        # Normalization code here (full version in chart)
        return sorted_results

class ReciprocalRankFusionRanking(RankingStrategy):
    """RRF ranking for combining multiple sources."""
    def __init__(self, k: int = 60): self.k = k
    def rank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for rank, result in enumerate(results, 1):
            result['rrf_score'] = 1.0 / (self.k + rank)
        return sorted(results, key=lambda x: x.get('rrf_score', 0.0), reverse=True)
