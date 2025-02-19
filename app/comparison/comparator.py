from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from app.models import ComparisonParams, ComparisonResult
from app.config import settings

class SignatureComparator:
    """Handles signature comparison using multiple feature types"""
    
    def __init__(self, params: Optional[ComparisonParams] = None):
        self.params = params or ComparisonParams()
    
    def compare_signatures(self, 
                          features1: Dict[str, np.ndarray],
                          features2: Dict[str, np.ndarray]) -> ComparisonResult:
        """
        Compare two signatures using their extracted features
        Returns a ComparisonResult with overall similarity and detailed scores
        """
        similarity_scores = {}
        
        # Calculate HOG feature similarity
        if self.params.hog_weight > 0 and 'hog' in features1 and 'hog' in features2:
            similarity_scores['hog'] = self._compare_hog_features(
                features1['hog'],
                features2['hog']
            ) * self.params.hog_weight
        
        # Calculate contour feature similarity
        if self.params.contour_weight > 0 and 'contour' in features1 and 'contour' in features2:
            similarity_scores['contour'] = self._compare_contour_features(
                features1['contour'],
                features2['contour']
            ) * self.params.contour_weight
        
        # Calculate density feature similarity
        if self.params.density_weight > 0 and 'density' in features1 and 'density' in features2:
            similarity_scores['density'] = self._compare_density_features(
                features1['density'],
                features2['density']
            ) * self.params.density_weight
        
        # Calculate overall similarity score
        total_weight = sum([
            self.params.hog_weight if 'hog' in similarity_scores else 0,
            self.params.contour_weight if 'contour' in similarity_scores else 0,
            self.params.density_weight if 'density' in similarity_scores else 0
        ])
        
        if total_weight == 0:
            overall_similarity = 0.0
        else:
            overall_similarity = sum(similarity_scores.values()) / total_weight
        
        return ComparisonResult(
            similarity_score=float(overall_similarity),
            is_match=overall_similarity >= self.params.threshold,
            details={k: float(v) for k, v in similarity_scores.items()}
        )

    def _compare_hog_features(self,
                            hog1: np.ndarray,
                            hog2: np.ndarray) -> float:
        """Compare HOG (Histogram of Oriented Gradients) features"""
        return float(cosine_similarity(
            hog1.reshape(1, -1),
            hog2.reshape(1, -1)
        )[0][0])

    def _compare_contour_features(self,
                                contour1: np.ndarray,
                                contour2: np.ndarray) -> float:
        """Compare contour features with length normalization"""
        # Normalize array lengths
        max_len = max(len(contour1), len(contour2))
        norm1 = np.pad(contour1, (0, max_len - len(contour1)))
        norm2 = np.pad(contour2, (0, max_len - len(contour2)))
        
        return float(cosine_similarity(
            norm1.reshape(1, -1),
            norm2.reshape(1, -1)
        )[0][0])

    def _compare_density_features(self,
                                density1: np.ndarray,
                                density2: np.ndarray) -> float:
        """Compare density grid features"""
        return float(cosine_similarity(
            density1.reshape(1, -1),
            density2.reshape(1, -1)
        )[0][0])

    @staticmethod
    def normalize_features(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize feature values to [0,1] range"""
        normalized = {}
        
        for feature_type, feature_array in features.items():
            if len(feature_array) == 0:
                continue
                
            min_val = np.min(feature_array)
            max_val = np.max(feature_array)
            
            if max_val - min_val > 0:
                normalized[feature_type] = (feature_array - min_val) / (max_val - min_val)
            else:
                normalized[feature_type] = feature_array
                
        return normalized

    @staticmethod
    def determine_threshold(signatures: List[Dict[str, np.ndarray]],
                          percentile: float = 90) -> float:
        """
        Determine optimal similarity threshold from a set of known signatures
        Uses the distribution of similarity scores to set threshold
        """
        if len(signatures) < 2:
            return settings.COMPARISON_DEFAULTS['threshold']
        
        similarities = []
        comparator = SignatureComparator()
        
        # Compare each signature with every other signature
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                result = comparator.compare_signatures(signatures[i], signatures[j])
                similarities.append(result.similarity_score)
        
        if not similarities:
            return settings.COMPARISON_DEFAULTS['threshold']
        
        # Set threshold at specified percentile
        return float(np.percentile(similarities, percentile))

def compare_signatures(features1: Dict[str, np.ndarray],
                      features2: Dict[str, np.ndarray],
                      params: Optional[ComparisonParams] = None) -> ComparisonResult:
    """
    Convenience function for one-off signature comparisons
    """
    comparator = SignatureComparator(params)
    return comparator.compare_signatures(features1, features2)