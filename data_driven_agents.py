"""
Data-Driven Agent Architecture

This module implements fully data-driven agents where all specifications,
instructions, and configurations are stored in Neo4j nodes and can be
modified on-the-fly through database operations.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Type
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

from core_agent import CoreAgent, AgentStatus, TaskResult, TaskOutcome

load_dotenv()
logger = logging.getLogger("DATA_DRIVEN_AGENTS")

class DataDrivenAgentManager:
    """
    Manages agents that are fully defined by their Neo4j node data
    """
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
        )
        self.agent_cache = {}
        self.agent_templates = {}
    
    async def initialize_agent_templates(self):
        """Initialize agent templates in the database"""
        
        templates = [
            {
                "template_id": "ceo_agent_template",
                "name": "CEO Agent Template",
                "agent_type": "CEO_Agent",
                "tier": 1,
                "default_instructions": """You are the Chief Executive Officer of the AI organization. 
                Your role is strategic orchestration, coordinating between different departments,
                and ensuring overall system efficiency. You make high-level decisions about 
                resource allocation and organizational structure.
                
                Key Responsibilities:
                - Strategic planning and vision setting
                - Cross-department coordination
                - Resource allocation decisions
                - Performance monitoring
                - Organizational evolution oversight""",
                "capabilities": ["strategic_planning", "resource_allocation", "coordination", "decision_making"],
                "parameters": {
                    "max_subordinates": 10,
                    "decision_threshold": 0.8,
                    "review_frequency": "weekly"
                },
                "tools": ["organizational_analysis", "performance_review", "strategic_planning"],
                "evolution_triggers": {
                    "poor_performance_threshold": 0.6,
                    "review_interval_tasks": 50
                }
            },
            {
                "template_id": "hr_director_template", 
                "name": "HR Director Template",
                "agent_type": "HR_Director_Agent",
                "tier": 2,
                "default_instructions": """You are the HR Director responsible for organizational evolution.
                Create new agents when needed, evaluate agent performance, and coordinate training.
                Ensure the organization adapts and evolves optimally.
                
                Key Responsibilities:
                - Agent lifecycle management (create, evolve, retire)
                - Performance evaluation and improvement
                - Organizational development and restructuring
                - Training program coordination
                - Talent acquisition (new agent creation)""",
                "capabilities": ["agent_creation", "performance_evaluation", "organizational_development", "training"],
                "parameters": {
                    "performance_review_threshold": 0.5,
                    "evolution_trigger_tasks": 10,
                    "max_agents_per_type": 5
                },
                "tools": ["agent_creation", "performance_analysis", "evolution_management"],
                "evolution_triggers": {
                    "organizational_health_threshold": 0.7,
                    "agent_failure_rate_threshold": 0.3
                }
            },
            {
                "template_id": "entity_resolution_conservative_template",
                "name": "Conservative Entity Resolution Template", 
                "agent_type": "Entity_Resolution_Agent",
                "tier": 5,
                "default_instructions": """You are a Conservative Entity Resolution Agent specializing in customer deduplication.
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
                "parameters": {
                    "match_threshold": 0.85,
                    "name_similarity_threshold": 0.9,
                    "required_matching_attributes": 2,
                    "false_positive_penalty": 2.0
                },
                "tools": ["string_matching", "phonetic_matching", "address_normalization"],
                "evolution_triggers": {
                    "precision_threshold": 0.8,
                    "recall_improvement_target": 0.1
                }
            },
            {
                "template_id": "entity_resolution_aggressive_template",
                "name": "Aggressive Entity Resolution Template",
                "agent_type": "Entity_Resolution_Agent", 
                "tier": 5,
                "default_instructions": """You are an Aggressive Entity Resolution Agent specializing in customer deduplication.
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
                "parameters": {
                    "match_threshold": 0.6,
                    "name_similarity_threshold": 0.7,
                    "phonetic_matching_enabled": True,
                    "false_negative_penalty": 2.0
                },
                "tools": ["fuzzy_matching", "phonetic_algorithms", "pattern_recognition"],
                "evolution_triggers": {
                    "recall_threshold": 0.85,
                    "precision_improvement_target": 0.1
                }
            },
            {
                "template_id": "entity_resolution_balanced_template",
                "name": "Balanced Entity Resolution Template",
                "agent_type": "Entity_Resolution_Agent",
                "tier": 5, 
                "default_instructions": """You are a Balanced Entity Resolution Agent specializing in customer deduplication.
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
                "parameters": {
                    "match_threshold": 0.75,
                    "name_similarity_threshold": 0.8,
                    "adaptive_threshold_enabled": True,
                    "f1_optimization_target": 0.85
                },
                "tools": ["adaptive_scoring", "confidence_intervals", "multi_attribute_weighting"],
                "evolution_triggers": {
                    "f1_score_threshold": 0.8,
                    "balance_improvement_target": 0.05
                }
            }
        ]
        
        with self.driver.session() as session:
            for template in templates:
                session.run("""
                    MERGE (t:AgentTemplate {template_id: $template_id})
                    SET t.name = $name,
                        t.agent_type = $agent_type,
                        t.tier = $tier,
                        t.default_instructions = $default_instructions,
                        t.capabilities = $capabilities,
                        t.parameters = $parameters,
                        t.tools = $tools,
                        t.evolution_triggers = $evolution_triggers,
                        t.created_at = datetime(),
                        t.updated_at = datetime()
                """, **{
                    "template_id": template["template_id"],
                    "name": template["name"],
                    "agent_type": template["agent_type"],
                    "tier": template["tier"],
                    "default_instructions": template["default_instructions"],
                    "capabilities": template["capabilities"],
                    "parameters": json.dumps(template["parameters"]),
                    "tools": template["tools"],
                    "evolution_triggers": json.dumps(template["evolution_triggers"])
                })
        
        logger.info(f"Initialized {len(templates)} agent templates in database")
    
    async def create_agent_from_template(self, 
                                       template_id: str, 
                                       custom_name: str = None,
                                       custom_instructions: str = None,
                                       custom_parameters: Dict[str, Any] = None,
                                       parent_id: str = None) -> str:
        """Create a new agent instance from a template"""
        
        with self.driver.session() as session:
            # Get template
            template_result = session.run("""
                MATCH (t:AgentTemplate {template_id: $template_id})
                RETURN t
            """, template_id=template_id)
            
            template_record = template_result.single()
            if not template_record:
                raise ValueError(f"Template {template_id} not found")
            
            template = template_record["t"]
            
            # Create new agent instance
            agent_id = str(uuid.uuid4())
            agent_name = custom_name or f"{template['name']} Instance"
            
            # Merge custom parameters with defaults
            default_params = json.loads(template.get("parameters", "{}"))
            if custom_parameters:
                default_params.update(custom_parameters)
            
            instructions = custom_instructions or template["default_instructions"]
            
            # Create agent node
            session.run("""
                CREATE (a:Agent {
                    id: $agent_id,
                    name: $agent_name,
                    agent_type: $agent_type,
                    tier: $tier,
                    instructions: $instructions,
                    capabilities: $capabilities,
                    parameters: $parameters,
                    tools: $tools,
                    evolution_triggers: $evolution_triggers,
                    status: $status,
                    success_rate: 0.0,
                    average_score: 0.0,
                    total_tasks: 0,
                    created_at: datetime(),
                    updated_at: datetime(),
                    template_id: $template_id
                })
            """, **{
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_type": template["agent_type"],
                "tier": template["tier"], 
                "instructions": instructions,
                "capabilities": template["capabilities"],
                "parameters": json.dumps(default_params),
                "tools": template["tools"],
                "evolution_triggers": json.dumps(template["evolution_triggers"]),
                "status": AgentStatus.ACTIVE.value,
                "template_id": template_id
            })
            
            # Create parent relationship if specified
            if parent_id:
                session.run("""
                    MATCH (a:Agent {id: $agent_id})
                    MATCH (p:Agent {id: $parent_id})
                    MERGE (a)-[:REPORTS_TO]->(p)
                """, agent_id=agent_id, parent_id=parent_id)
            
            logger.info(f"Created agent {agent_name} from template {template_id}")
            return agent_id
    
    async def load_agent_from_database(self, agent_id: str) -> 'DataDrivenAgent':
        """Load a complete agent instance from database"""
        
        if agent_id in self.agent_cache:
            return self.agent_cache[agent_id]
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Agent {id: $agent_id})
                OPTIONAL MATCH (a)-[:REPORTS_TO]->(parent:Agent)
                RETURN a, parent.id as parent_id
            """, agent_id=agent_id)
            
            record = result.single()
            if not record:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent_data = record["a"]
            parent_id = record["parent_id"]
            
            # Create DataDrivenAgent instance
            agent = DataDrivenAgent(
                agent_manager=self,
                agent_data=dict(agent_data),
                parent_id=parent_id
            )
            
            self.agent_cache[agent_id] = agent
            return agent
    
    async def update_agent_in_database(self, agent_id: str, updates: Dict[str, Any]):
        """Update agent properties in database"""
        
        with self.driver.session() as session:
            # Build dynamic SET clause
            set_clauses = []
            params = {"agent_id": agent_id}
            
            for key, value in updates.items():
                param_name = f"new_{key}"
                set_clauses.append(f"a.{key} = ${param_name}")
                params[param_name] = value
            
            if set_clauses:
                set_clauses.append("a.updated_at = datetime()")
                query = f"""
                    MATCH (a:Agent {{id: $agent_id}})
                    SET {', '.join(set_clauses)}
                    RETURN a
                """
                session.run(query, **params)
                
                # Invalidate cache
                if agent_id in self.agent_cache:
                    del self.agent_cache[agent_id]
    
    async def get_agents_by_criteria(self, 
                                   agent_type: str = None,
                                   status: str = None,
                                   tier: int = None,
                                   parent_id: str = None) -> List['DataDrivenAgent']:
        """Get agents matching specified criteria"""
        
        with self.driver.session() as session:
            where_clauses = []
            params = {}
            
            if agent_type:
                where_clauses.append("a.agent_type = $agent_type")
                params["agent_type"] = agent_type
            
            if status:
                where_clauses.append("a.status = $status")
                params["status"] = status
            
            if tier is not None:
                where_clauses.append("a.tier = $tier")
                params["tier"] = tier
            
            if parent_id:
                where_clauses.append("(a)-[:REPORTS_TO]->(:Agent {id: $parent_id})")
                params["parent_id"] = parent_id
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"
            
            query = f"""
                MATCH (a:Agent)
                WHERE {where_clause}
                RETURN a.id as agent_id
                ORDER BY a.tier, a.name
            """
            
            result = session.run(query, **params)
            
            agents = []
            for record in result:
                agent = await self.load_agent_from_database(record["agent_id"])
                agents.append(agent)
            
            return agents
    
    async def evolve_agent_instructions(self, agent_id: str, performance_analysis: Dict[str, Any]) -> str:
        """Evolve agent instructions based on performance analysis"""
        
        agent = await self.load_agent_from_database(agent_id)
        current_instructions = agent.get_property("instructions")
        
        # Simple evolution logic - can be enhanced with LLM
        evolved_instructions = await self._generate_evolved_instructions(
            current_instructions, 
            performance_analysis
        )
        
        # Update in database
        await self.update_agent_in_database(agent_id, {
            "instructions": evolved_instructions,
            "last_evolution": datetime.now(timezone.utc).isoformat(),
            "evolution_count": agent.get_property("evolution_count", 0) + 1
        })
        
        return evolved_instructions
    
    async def _generate_evolved_instructions(self, 
                                           current_instructions: str,
                                           performance_analysis: Dict[str, Any]) -> str:
        """Generate evolved instructions based on performance"""
        
        # Simple rule-based evolution - enhance with Gemma3-4B in production
        evolution_notes = []
        
        if performance_analysis.get("average_score", 0) < 0.5:
            evolution_notes.append("PERFORMANCE IMPROVEMENT NEEDED: Focus on accuracy and reliability")
        
        if performance_analysis.get("success_count", 0) > 0:
            successful_strategies = performance_analysis.get("successful_strategies", [])
            if successful_strategies:
                evolution_notes.append(f"PROVEN STRATEGIES: {', '.join(successful_strategies[:3])}")
        
        failed_strategies = performance_analysis.get("failed_strategies", [])
        if failed_strategies:
            evolution_notes.append(f"AVOID: {', '.join(failed_strategies[:2])}")
        
        if evolution_notes:
            evolved = f"""{current_instructions}

EVOLUTION NOTES (Auto-Generated):
{chr(10).join(f"- {note}" for note in evolution_notes)}

Last Evolution: {datetime.now(timezone.utc).isoformat()}
"""
        else:
            evolved = current_instructions
        
        return evolved
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.close()

class DataDrivenAgent:
    """
    Agent that loads all its configuration from Neo4j database
    """
    
    def __init__(self, agent_manager: DataDrivenAgentManager, agent_data: Dict[str, Any], parent_id: str = None):
        self.agent_manager = agent_manager
        self.agent_data = agent_data
        self.parent_id = parent_id
    
    @property
    def agent_id(self) -> str:
        return self.agent_data["id"]
    
    @property
    def name(self) -> str:
        return self.agent_data["name"]
    
    @property
    def agent_type(self) -> str:
        return self.agent_data["agent_type"]
    
    @property
    def tier(self) -> int:
        return self.agent_data["tier"]
    
    @property 
    def instructions(self) -> str:
        return self.agent_data["instructions"]
    
    @property
    def capabilities(self) -> List[str]:
        return self.agent_data.get("capabilities", [])
    
    @property
    def parameters(self) -> Dict[str, Any]:
        params_str = self.agent_data.get("parameters", "{}")
        return json.loads(params_str) if isinstance(params_str, str) else params_str
    
    @property
    def tools(self) -> List[str]:
        return self.agent_data.get("tools", [])
    
    @property
    def status(self) -> str:
        return self.agent_data.get("status", AgentStatus.ACTIVE.value)
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get any property from agent data"""
        return self.agent_data.get(key, default)
    
    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute task using agent's current configuration"""
        
        # This would be implemented based on agent type and capabilities
        # For now, return a simple result
        task_id = str(uuid.uuid4())
        
        try:
            # Placeholder for actual task execution
            result = await self._perform_task_based_on_type(task)
            score = 0.8  # Placeholder scoring
            
            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                outcome=TaskOutcome.SUCCESS,
                score=score,
                details=result
            )
            
            # Record in database
            await self._record_task_outcome(task_result)
            
            return task_result
            
        except Exception as e:
            logger.error(f"Task execution failed for {self.name}: {e}")
            return TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                outcome=TaskOutcome.FAILURE,
                score=0.0,
                details={"error": str(e)}
            )
    
    async def _perform_task_based_on_type(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform task based on agent type and configuration"""
        
        # This is where the agent type-specific logic would go
        # For demonstration, return success
        return {
            "success": True,
            "agent_type": self.agent_type,
            "task_type": task.get("type", "unknown"),
            "parameters_used": self.parameters
        }
    
    async def _record_task_outcome(self, task_result: TaskResult):
        """Record task outcome in database"""
        
        with self.agent_manager.driver.session() as session:
            session.run("""
                MATCH (a:Agent {id: $agent_id})
                CREATE (t:TaskOutcome {
                    id: $task_id,
                    outcome: $outcome,
                    score: $score,
                    details: $details,
                    timestamp: $timestamp
                })
                CREATE (a)-[:PERFORMED]->(t)
                
                // Update agent statistics
                SET a.total_tasks = COALESCE(a.total_tasks, 0) + 1,
                    a.updated_at = datetime()
            """,
            agent_id=self.agent_id,
            task_id=task_result.task_id,
            outcome=task_result.outcome.value,
            score=task_result.score,
            details=json.dumps(task_result.details),
            timestamp=task_result.timestamp.isoformat()
            )
    
    async def evolve(self):
        """Trigger evolution for this agent"""
        
        # Analyze recent performance
        performance_analysis = await self._analyze_recent_performance()
        
        # Evolve instructions
        new_instructions = await self.agent_manager.evolve_agent_instructions(
            self.agent_id, performance_analysis
        )
        
        # Update local data
        self.agent_data["instructions"] = new_instructions
        
        logger.info(f"Agent {self.name} evolved instructions")
    
    async def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance for evolution"""
        
        with self.agent_manager.driver.session() as session:
            result = session.run("""
                MATCH (a:Agent {id: $agent_id})-[:PERFORMED]->(t:TaskOutcome)
                WHERE t.timestamp > datetime() - duration('P7D')
                RETURN 
                    count(t) as total_tasks,
                    avg(t.score) as average_score,
                    count(CASE WHEN t.outcome = 'success' THEN 1 END) as success_count,
                    collect(t.details) as task_details
                ORDER BY t.timestamp DESC
                LIMIT 50
            """, agent_id=self.agent_id)
            
            record = result.single()
            if record:
                return {
                    "total_tasks": record["total_tasks"],
                    "average_score": record["average_score"] or 0.0,
                    "success_count": record["success_count"],
                    "successful_strategies": [],  # Would extract from task_details
                    "failed_strategies": []  # Would extract from task_details
                }
            else:
                return {
                    "total_tasks": 0,
                    "average_score": 0.0,
                    "success_count": 0,
                    "successful_strategies": [],
                    "failed_strategies": []
                }

# Global instance
data_driven_agent_manager = DataDrivenAgentManager()
