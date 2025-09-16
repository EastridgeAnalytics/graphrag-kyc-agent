"""
HR Director Agent - Responsible for organizational evolution and agent lifecycle management

This agent can create new agents, evaluate performance, and evolve the organizational structure.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from core_agent import CoreAgent, AgentStatus, TaskResult, TaskOutcome, orchestrator

logger = logging.getLogger("HR_DIRECTOR")

class HRDirectorAgent(CoreAgent):
    """
    HR Director specializes in organizational management and agent lifecycle
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_type="HR_Director_Agent",
            name="HR Director Agent",
            **kwargs
        )
    
    async def _perform_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HR-related tasks like agent creation, performance review, etc.
        """
        task_type = task.get("type", "unknown")
        
        if task_type == "create_agent":
            return await self._create_agent(task)
        elif task_type == "evaluate_performance":
            return await self._evaluate_agent_performance(task)
        elif task_type == "evolution_review":
            return await self._conduct_evolution_review(task)
        elif task_type == "replace_poor_performer":
            return await self._replace_poor_performer(task)
        else:
            return {"success": False, "error": f"Unknown task type: {task_type}"}
    
    async def _create_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new agent based on specifications
        """
        try:
            agent_spec = task.get("agent_spec", {})
            
            # Validate required fields
            required_fields = ["name", "agent_type", "instructions"]
            for field in required_fields:
                if field not in agent_spec:
                    return {"success": False, "error": f"Missing required field: {field}"}
            
            # Create the new agent
            new_agent = CoreAgent(
                name=agent_spec["name"],
                agent_type=agent_spec["agent_type"],
                tier=agent_spec.get("tier", 5),
                instructions=agent_spec["instructions"],
                capabilities=agent_spec.get("capabilities", []),
                parent_id=agent_spec.get("parent_id", self.agent_id)
            )
            
            # Add to orchestrator
            orchestrator.active_agents[new_agent.agent_id] = new_agent
            
            logger.info(f"HR Director created new agent: {new_agent.name} ({new_agent.agent_id})")
            
            return {
                "success": True,
                "agent_id": new_agent.agent_id,
                "agent_name": new_agent.name,
                "message": f"Successfully created {new_agent.name}"
            }
            
        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _evaluate_agent_performance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the performance of specified agents
        """
        try:
            agent_ids = task.get("agent_ids", [])
            evaluations = []
            
            for agent_id in agent_ids:
                agent = await orchestrator.get_agent(agent_id)
                if not agent:
                    continue
                
                evaluation = {
                    "agent_id": agent_id,
                    "agent_name": agent.name,
                    "success_rate": agent.performance_metrics.success_rate,
                    "average_score": agent.performance_metrics.average_score,
                    "total_tasks": agent.performance_metrics.total_tasks,
                    "status": agent.status.value,
                    "recommendation": self._get_performance_recommendation(agent)
                }
                evaluations.append(evaluation)
            
            return {
                "success": True,
                "evaluations": evaluations,
                "summary": self._generate_performance_summary(evaluations)
            }
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_performance_recommendation(self, agent: CoreAgent) -> str:
        """Generate performance recommendation for an agent"""
        if agent.performance_metrics.success_rate > 0.8:
            return "excellent_performer"
        elif agent.performance_metrics.success_rate > 0.6:
            return "good_performer"
        elif agent.performance_metrics.success_rate > 0.4:
            return "needs_improvement"
        else:
            return "replace_candidate"
    
    def _generate_performance_summary(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """Generate a summary of performance evaluations"""
        if not evaluations:
            return {"total_agents": 0}
        
        total_agents = len(evaluations)
        excellent_count = len([e for e in evaluations if e["recommendation"] == "excellent_performer"])
        good_count = len([e for e in evaluations if e["recommendation"] == "good_performer"])
        needs_improvement_count = len([e for e in evaluations if e["recommendation"] == "needs_improvement"])
        replace_count = len([e for e in evaluations if e["recommendation"] == "replace_candidate"])
        
        avg_success_rate = sum(e["success_rate"] for e in evaluations) / total_agents
        
        return {
            "total_agents": total_agents,
            "excellent_performers": excellent_count,
            "good_performers": good_count,
            "needs_improvement": needs_improvement_count,
            "replace_candidates": replace_count,
            "average_success_rate": avg_success_rate,
            "organizational_health": "healthy" if avg_success_rate > 0.7 else "needs_attention"
        }
    
    async def _conduct_evolution_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct a comprehensive review to determine if organizational evolution is needed
        """
        try:
            # Get all active agents
            agent_type = task.get("agent_type", "Entity_Resolution_Agent")
            agents = await orchestrator.get_agents_by_type(agent_type)
            
            if not agents:
                return {"success": True, "message": "No agents to review", "action_needed": False}
            
            # Evaluate their collective performance
            total_score = sum(agent.performance_metrics.average_score for agent in agents)
            avg_performance = total_score / len(agents) if agents else 0
            
            poor_performers = [agent for agent in agents if agent.performance_metrics.success_rate < 0.6]
            excellent_performers = [agent for agent in agents if agent.performance_metrics.success_rate > 0.8]
            
            evolution_needed = len(poor_performers) > 0 or avg_performance < 0.65
            
            recommendations = []
            if evolution_needed:
                if poor_performers:
                    recommendations.append({
                        "action": "replace_poor_performers",
                        "agents": [agent.agent_id for agent in poor_performers],
                        "reason": "Performance below threshold"
                    })
                
                if excellent_performers:
                    recommendations.append({
                        "action": "clone_successful_strategies",
                        "source_agents": [agent.agent_id for agent in excellent_performers],
                        "reason": "Replicate successful patterns"
                    })
            
            return {
                "success": True,
                "evolution_needed": evolution_needed,
                "average_performance": avg_performance,
                "poor_performers_count": len(poor_performers),
                "excellent_performers_count": len(excellent_performers),
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Evolution review failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _replace_poor_performer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace a poor-performing agent with an evolved version
        """
        try:
            old_agent_id = task.get("old_agent_id")
            successful_agent_ids = task.get("successful_agent_ids", [])
            
            if not old_agent_id:
                return {"success": False, "error": "old_agent_id is required"}
            
            # Get the old agent
            old_agent = await orchestrator.get_agent(old_agent_id)
            if not old_agent:
                return {"success": False, "error": "Old agent not found"}
            
            # Analyze successful strategies from other agents
            successful_strategies = []
            if successful_agent_ids:
                for agent_id in successful_agent_ids:
                    successful_agent = await orchestrator.get_agent(agent_id)
                    if successful_agent:
                        # Extract successful patterns from their instructions
                        successful_strategies.append(successful_agent.instructions)
            
            # Create evolved instructions
            evolved_instructions = await self._create_evolved_instructions(
                old_agent.instructions,
                successful_strategies
            )
            
            # Create new agent with evolved capabilities
            new_agent_spec = {
                "name": f"Evolved_{old_agent.name}",
                "agent_type": old_agent.agent_type,
                "tier": old_agent.tier,
                "instructions": evolved_instructions,
                "capabilities": old_agent.capabilities + ["evolved_learning"],
                "parent_id": old_agent.parent_id
            }
            
            # Create the new agent
            creation_result = await self._create_agent({"agent_spec": new_agent_spec})
            
            if creation_result["success"]:
                # Deprecate old agent
                old_agent.status = AgentStatus.DEPRECATED
                await old_agent._update_performance_in_db()
                
                # Remove from active agents
                if old_agent_id in orchestrator.active_agents:
                    del orchestrator.active_agents[old_agent_id]
                
                logger.info(f"Replaced {old_agent.name} with evolved version")
                
                return {
                    "success": True,
                    "old_agent_id": old_agent_id,
                    "new_agent_id": creation_result["agent_id"],
                    "message": f"Successfully replaced {old_agent.name} with evolved version"
                }
            else:
                return creation_result
                
        except Exception as e:
            logger.error(f"Agent replacement failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_evolved_instructions(self, 
                                         old_instructions: str, 
                                         successful_strategies: List[str]) -> str:
        """
        Create evolved instructions by combining old approach with successful strategies
        """
        base_instructions = old_instructions
        
        if successful_strategies:
            # Simple strategy combination - in production, use LLM for better results
            strategy_summary = "Key successful approaches from high performers:\n"
            for i, strategy in enumerate(successful_strategies[:2], 1):
                # Extract key phrases from successful strategies
                key_points = strategy.split('.')[:3]  # First few sentences
                strategy_summary += f"{i}. {'. '.join(key_points)}\n"
            
            evolved_instructions = f"""{base_instructions}

EVOLVED LEARNING - Incorporated successful strategies:
{strategy_summary}

Apply these proven approaches while maintaining your core capabilities."""
        else:
            evolved_instructions = f"""{base_instructions}

EVOLVED LEARNING - Enhanced with adaptive approach:
- Focus on incremental improvements
- Learn from each task outcome
- Adapt strategy based on performance feedback"""
        
        return evolved_instructions
    
    async def spawn_entity_resolution_agents(self) -> List[str]:
        """
        Spawn three Entity Resolution Agents with different strategies for the test scenario
        """
        logger.info("HR Director spawning Entity Resolution Agents")
        
        agent_specs = [
            {
                "name": "Conservative Matcher",
                "agent_type": "Entity_Resolution_Agent", 
                "tier": 5,
                "instructions": """You are a Conservative Entity Resolution Agent specializing in customer deduplication.
                Your strategy prioritizes precision over recall - you only match entities when you're highly confident.
                
                Matching Strategy:
                - Use strict thresholds for name similarity (>90% match)
                - Require at least 2 matching attributes (name + phone/address)
                - Weight exact matches heavily
                - Be conservative with fuzzy matching
                - Focus on avoiding false positives
                
                When analyzing customer records:
                1. Compare names using exact and near-exact matching
                2. Verify phone numbers and addresses for confirmation
                3. Only declare a match when confidence > 85%""",
                "capabilities": ["entity_resolution", "conservative_matching", "precision_focused"],
                "parent_id": self.agent_id
            },
            {
                "name": "Aggressive Matcher", 
                "agent_type": "Entity_Resolution_Agent",
                "tier": 5,
                "instructions": """You are an Aggressive Entity Resolution Agent specializing in customer deduplication.
                Your strategy prioritizes recall over precision - you aim to find all possible matches.
                
                Matching Strategy:
                - Use lower thresholds for name similarity (>70% match)
                - Accept single strong attribute matches
                - Use advanced fuzzy matching and phonetic algorithms
                - Consider partial address and phone matches
                - Focus on maximizing match detection
                
                When analyzing customer records:
                1. Apply fuzzy string matching with multiple algorithms
                2. Use phonetic matching for names (Soundex, Metaphone)
                3. Accept matches when confidence > 60%
                4. Consider variations in formatting and abbreviations""",
                "capabilities": ["entity_resolution", "aggressive_matching", "recall_focused"],
                "parent_id": self.agent_id
            },
            {
                "name": "Balanced Matcher",
                "agent_type": "Entity_Resolution_Agent", 
                "tier": 5,
                "instructions": """You are a Balanced Entity Resolution Agent specializing in customer deduplication.
                Your strategy seeks optimal balance between precision and recall.
                
                Matching Strategy:
                - Use moderate thresholds for name similarity (>80% match)
                - Weight different attributes based on reliability
                - Apply machine learning-inspired scoring
                - Use adaptive thresholds based on data quality
                - Balance false positives and false negatives
                
                When analyzing customer records:
                1. Compute weighted similarity scores across all attributes
                2. Apply different thresholds for different attribute combinations
                3. Use confidence intervals for decision making
                4. Declare matches when confidence is 70-75%""",
                "capabilities": ["entity_resolution", "balanced_matching", "adaptive_scoring"],
                "parent_id": self.agent_id
            }
        ]
        
        created_agents = []
        for spec in agent_specs:
            result = await self._create_agent({"agent_spec": spec})
            if result["success"]:
                created_agents.append(result["agent_id"])
        
        logger.info(f"HR Director created {len(created_agents)} Entity Resolution Agents")
        return created_agents
