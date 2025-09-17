"""
Entity Resolution Agents - Specialized agents for customer deduplication

These agents implement different matching strategies to demonstrate self-evolution
and performance comparison in the customer deduplication challenge.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import uuid
from datetime import datetime
import difflib
import re
from core_agent import CoreAgent, TaskResult, TaskOutcome
import jellyfish  # For phonetic matching - you may need to install: pip install jellyfish
import recordlinkage as rl  # For record linkage - you may need to install: pip install recordlinkage

logger = logging.getLogger("ENTITY_RESOLUTION")

class EntityResolutionAgent(CoreAgent):
    """
    Base class for entity resolution agents with common functionality
    """
    
    def __init__(self, matching_strategy: str = "balanced", **kwargs):
        super().__init__(**kwargs)
        self.matching_strategy = matching_strategy
        self.match_threshold = 0.75  # Default threshold
        self.statistics = {
            "total_comparisons": 0,
            "matches_found": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
    
    async def _perform_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute entity resolution tasks
        """
        task_type = task.get("type", "deduplicate")
        
        if task_type == "deduplicate":
            return await self._deduplicate_customers(task)
        elif task_type == "compare_records":
            return await self._compare_records(task)
        else:
            return {"success": False, "error": f"Unknown task type: {task_type}"}
    
    async def _deduplicate_customers(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main deduplication logic - find duplicate customer records
        """
        try:
            customer_records = task.get("customer_records", [])
            if not customer_records:
                return {"success": False, "error": "No customer records provided"}
            
            # Find duplicate groups
            duplicate_groups = await self._find_duplicate_groups(customer_records)
            
            # Calculate performance metrics if ground truth is provided
            ground_truth = task.get("ground_truth", None)
            performance_metrics = None
            if ground_truth:
                performance_metrics = self._calculate_performance(duplicate_groups, ground_truth)
            
            return {
                "success": True,
                "duplicate_groups": duplicate_groups,
                "total_groups": len(duplicate_groups),
                "total_records_processed": len(customer_records),
                "matching_strategy": self.matching_strategy,
                "performance_metrics": performance_metrics,
                "agent_statistics": self.statistics
            }
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _find_duplicate_groups(self, customer_records: List[Dict]) -> List[List[Dict]]:
        """
        Find groups of duplicate customer records
        """
        duplicate_groups = []
        processed_ids = set()
        
        for i, record1 in enumerate(customer_records):
            if record1.get("id") in processed_ids:
                continue
            
            current_group = [record1]
            processed_ids.add(record1.get("id"))
            
            # Compare with remaining records
            for j, record2 in enumerate(customer_records[i+1:], i+1):
                if record2.get("id") in processed_ids:
                    continue
                
                similarity_score = await self._calculate_similarity(record1, record2)
                self.statistics["total_comparisons"] += 1
                
                if similarity_score >= self.match_threshold:
                    current_group.append(record2)
                    processed_ids.add(record2.get("id"))
                    self.statistics["matches_found"] += 1
            
            # Only add groups with duplicates
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
        
        return duplicate_groups
    
    async def _calculate_similarity(self, record1: Dict, record2: Dict) -> float:
        """
        Calculate similarity between two customer records
        Override this method in subclasses for different strategies
        """
        # Base implementation - to be overridden
        name_sim = self._name_similarity(record1.get("name", ""), record2.get("name", ""))
        phone_sim = self._phone_similarity(record1.get("phone", ""), record2.get("phone", ""))
        address_sim = self._address_similarity(record1.get("address", ""), record2.get("address", ""))
        
        # Weighted average
        total_score = (name_sim * 0.5) + (phone_sim * 0.25) + (address_sim * 0.25)
        return total_score
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity"""
        if not name1 or not name2:
            return 0.0
        
        # Normalize names
        name1_clean = self._normalize_name(name1)
        name2_clean = self._normalize_name(name2)
        
        # Exact match
        if name1_clean == name2_clean:
            return 1.0
        
        # Sequence matching
        seq_sim = difflib.SequenceMatcher(None, name1_clean, name2_clean).ratio()
        return seq_sim
    
    def _phone_similarity(self, phone1: str, phone2: str) -> float:
        """Calculate phone number similarity"""
        if not phone1 or not phone2:
            return 0.0
        
        # Normalize phone numbers (remove non-digits)
        phone1_clean = re.sub(r'\D', '', phone1)
        phone2_clean = re.sub(r'\D', '', phone2)
        
        if phone1_clean == phone2_clean:
            return 1.0
        
        # Check if one is contained in the other (for different formats)
        if phone1_clean in phone2_clean or phone2_clean in phone1_clean:
            return 0.8
        
        return 0.0
    
    def _address_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate address similarity"""
        if not addr1 or not addr2:
            return 0.0
        
        # Normalize addresses
        addr1_clean = self._normalize_address(addr1)
        addr2_clean = self._normalize_address(addr2)
        
        if addr1_clean == addr2_clean:
            return 1.0
        
        # Sequence matching for partial matches
        seq_sim = difflib.SequenceMatcher(None, addr1_clean, addr2_clean).ratio()
        return seq_sim
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison"""
        return re.sub(r'\s+', ' ', name.lower().strip())
    
    def _normalize_address(self, address: str) -> str:
        """Normalize address for comparison"""
        # Common address abbreviations
        address = address.lower()
        address = re.sub(r'\bstreet\b', 'st', address)
        address = re.sub(r'\bavenue\b', 'ave', address)
        address = re.sub(r'\broad\b', 'rd', address)
        address = re.sub(r'\bdrive\b', 'dr', address)
        address = re.sub(r'\blane\b', 'ln', address)
        return re.sub(r'\s+', ' ', address.strip())
    
    def _calculate_performance(self, found_groups: List[List[Dict]], ground_truth: List[List[str]]) -> Dict:
        """Calculate precision, recall, and F1 score"""
        # Convert found groups to set of pairs for comparison
        found_pairs = set()
        for group in found_groups:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    id1, id2 = group[i]["id"], group[j]["id"]
                    found_pairs.add(tuple(sorted([id1, id2])))
        
        # Convert ground truth to set of pairs
        true_pairs = set()
        for group in ground_truth:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    id1, id2 = group[i], group[j]
                    true_pairs.add(tuple(sorted([id1, id2])))
        
        # Calculate metrics
        true_positives = len(found_pairs & true_pairs)
        false_positives = len(found_pairs - true_pairs)
        false_negatives = len(true_pairs - found_pairs)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Update agent statistics
        self.statistics.update({
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        })
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

class ConservativeMatcherAgent(EntityResolutionAgent):
    """
    Conservative matching strategy - prioritizes precision over recall
    """
    
    def __init__(self, **kwargs):
        super().__init__(matching_strategy="conservative", **kwargs)
        self.match_threshold = 0.85  # High threshold for conservative matching
    
    async def _calculate_similarity(self, record1: Dict, record2: Dict) -> float:
        """
        Conservative similarity calculation - requires high confidence
        """
        name_sim = self._conservative_name_similarity(record1.get("name", ""), record2.get("name", ""))
        phone_sim = self._phone_similarity(record1.get("phone", ""), record2.get("phone", ""))
        address_sim = self._address_similarity(record1.get("address", ""), record2.get("address", ""))
        
        # Require at least 2 strong matches
        high_confidence_matches = sum([
            name_sim > 0.9,
            phone_sim > 0.8,
            address_sim > 0.8
        ])
        
        if high_confidence_matches < 2:
            return 0.0  # Not confident enough
        
        # Weighted heavily towards exact matches
        total_score = (name_sim * 0.6) + (phone_sim * 0.2) + (address_sim * 0.2)
        return total_score
    
    def _conservative_name_similarity(self, name1: str, name2: str) -> float:
        """More strict name matching"""
        if not name1 or not name2:
            return 0.0
        
        name1_clean = self._normalize_name(name1)
        name2_clean = self._normalize_name(name2)
        
        # Exact match gets full score
        if name1_clean == name2_clean:
            return 1.0
        
        # Very high threshold for fuzzy matching
        seq_sim = difflib.SequenceMatcher(None, name1_clean, name2_clean).ratio()
        return seq_sim if seq_sim > 0.95 else 0.0

class AggressiveMatcherAgent(EntityResolutionAgent):
    """
    Aggressive matching strategy - prioritizes recall over precision
    """
    
    def __init__(self, **kwargs):
        super().__init__(matching_strategy="aggressive", **kwargs)
        self.match_threshold = 0.60  # Lower threshold for aggressive matching
    
    async def _calculate_similarity(self, record1: Dict, record2: Dict) -> float:
        """
        Aggressive similarity calculation - more permissive
        """
        name_sim = self._aggressive_name_similarity(record1.get("name", ""), record2.get("name", ""))
        phone_sim = self._phone_similarity(record1.get("phone", ""), record2.get("phone", ""))
        address_sim = self._aggressive_address_similarity(record1.get("address", ""), record2.get("address", ""))
        
        # Accept single strong match or multiple weak matches
        max_single_sim = max(name_sim, phone_sim, address_sim)
        if max_single_sim > 0.8:
            return max_single_sim
        
        # Weighted average with bonus for multiple matches
        total_score = (name_sim * 0.4) + (phone_sim * 0.3) + (address_sim * 0.3)
        
        # Bonus for multiple weak matches
        match_count = sum([name_sim > 0.5, phone_sim > 0.5, address_sim > 0.5])
        if match_count >= 2:
            total_score += 0.1
        
        return total_score
    
    def _aggressive_name_similarity(self, name1: str, name2: str) -> float:
        """More permissive name matching with phonetic algorithms"""
        if not name1 or not name2:
            return 0.0
        
        name1_clean = self._normalize_name(name1)
        name2_clean = self._normalize_name(name2)
        
        # Exact match
        if name1_clean == name2_clean:
            return 1.0
        
        # Sequence matching
        seq_sim = difflib.SequenceMatcher(None, name1_clean, name2_clean).ratio()
        
        # Phonetic matching
        try:
            soundex_sim = 1.0 if jellyfish.soundex(name1_clean) == jellyfish.soundex(name2_clean) else 0.0
            metaphone_sim = 1.0 if jellyfish.metaphone(name1_clean) == jellyfish.metaphone(name2_clean) else 0.0
            
            # Take best of the three methods
            return max(seq_sim, soundex_sim * 0.8, metaphone_sim * 0.8)
        except:
            return seq_sim
    
    def _aggressive_address_similarity(self, addr1: str, addr2: str) -> float:
        """More permissive address matching"""
        if not addr1 or not addr2:
            return 0.0
        
        addr1_clean = self._normalize_address(addr1)
        addr2_clean = self._normalize_address(addr2)
        
        if addr1_clean == addr2_clean:
            return 1.0
        
        # Check for partial matches
        addr1_tokens = set(addr1_clean.split())
        addr2_tokens = set(addr2_clean.split())
        
        if len(addr1_tokens) == 0 or len(addr2_tokens) == 0:
            return 0.0
        
        # Jaccard similarity of tokens
        intersection = len(addr1_tokens & addr2_tokens)
        union = len(addr1_tokens | addr2_tokens)
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Also check sequence similarity
        seq_sim = difflib.SequenceMatcher(None, addr1_clean, addr2_clean).ratio()
        
        return max(jaccard_sim, seq_sim)

class BalancedMatcherAgent(EntityResolutionAgent):
    """
    Balanced matching strategy - optimizes F1 score
    """
    
    def __init__(self, **kwargs):
        super().__init__(matching_strategy="balanced", **kwargs)
        self.match_threshold = 0.75  # Moderate threshold
        self.adaptive_threshold = True
    
    async def _calculate_similarity(self, record1: Dict, record2: Dict) -> float:
        """
        Balanced similarity calculation with adaptive scoring
        """
        name_sim = self._balanced_name_similarity(record1.get("name", ""), record2.get("name", ""))
        phone_sim = self._phone_similarity(record1.get("phone", ""), record2.get("phone", ""))
        address_sim = self._address_similarity(record1.get("address", ""), record2.get("address", ""))
        
        # Adaptive weighting based on data quality
        weights = self._calculate_adaptive_weights(record1, record2)
        
        total_score = (name_sim * weights["name"]) + (phone_sim * weights["phone"]) + (address_sim * weights["address"])
        
        # Apply confidence interval adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(name_sim, phone_sim, address_sim)
        
        return total_score * confidence_adjustment
    
    def _balanced_name_similarity(self, name1: str, name2: str) -> float:
        """Balanced name matching with multiple techniques"""
        if not name1 or not name2:
            return 0.0
        
        name1_clean = self._normalize_name(name1)
        name2_clean = self._normalize_name(name2)
        
        if name1_clean == name2_clean:
            return 1.0
        
        # Multiple similarity measures
        seq_sim = difflib.SequenceMatcher(None, name1_clean, name2_clean).ratio()
        
        # Token-based similarity for multi-word names
        tokens1 = set(name1_clean.split())
        tokens2 = set(name2_clean.split())
        
        if tokens1 and tokens2:
            token_sim = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        else:
            token_sim = 0.0
        
        # Combine measures
        return max(seq_sim, token_sim * 0.9)
    
    def _calculate_adaptive_weights(self, record1: Dict, record2: Dict) -> Dict[str, float]:
        """Calculate adaptive weights based on data quality"""
        # Base weights
        weights = {"name": 0.5, "phone": 0.25, "address": 0.25}
        
        # Adjust based on data completeness and quality
        if not record1.get("phone") or not record2.get("phone"):
            weights["name"] += 0.125
            weights["address"] += 0.125
            weights["phone"] = 0.0
        
        if not record1.get("address") or not record2.get("address"):
            weights["name"] += 0.125
            weights["phone"] += 0.125
            weights["address"] = 0.0
        
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_confidence_adjustment(self, name_sim: float, phone_sim: float, address_sim: float) -> float:
        """Calculate confidence adjustment factor"""
        scores = [s for s in [name_sim, phone_sim, address_sim] if s > 0]
        
        if not scores:
            return 0.5
        
        # Higher confidence when multiple scores agree
        score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
        
        # Lower variance = higher confidence
        confidence = 1.0 - min(score_variance, 0.2)  # Cap the penalty
        
        return confidence

# Factory function to create agents
def create_entity_resolution_agent(strategy: str, **kwargs) -> EntityResolutionAgent:
    """Factory function to create entity resolution agents"""
    if strategy == "conservative":
        return ConservativeMatcherAgent(**kwargs)
    elif strategy == "aggressive":
        return AggressiveMatcherAgent(**kwargs)
    elif strategy == "balanced":
        return BalancedMatcherAgent(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
