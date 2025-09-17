"""
Core Agent Framework for Self-Evolving AI Infrastructure

This module implements the foundational architecture where agents exist as Neo4j nodes
capable of self-organization, learning, and evolution.
"""

import os
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel
from neo4j import GraphDatabase
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CORE_AGENT")

# Load environment variables
load_dotenv()

# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Global driver instance
driver = get_neo4j_driver()

class AgentStatus(Enum):
    ACTIVE = "active"
    LEARNING = "learning" 
    EVOLVING = "evolving"
    DEPRECATED = "deprecated"
    SPAWNING = "spawning"

class TaskOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"

@dataclass
class TaskResult:
    """Represents the result of an agent task execution"""
    task_id: str
    agent_id: str
    outcome: TaskOutcome
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
class PerformanceMetrics(BaseModel):
    """Agent performance tracking"""
    success_rate: float = 0.0
    average_score: float = 0.0
    total_tasks: int = 0
    recent_performance: List[float] = []
    last_evolution: Optional[datetime] = None

class AgentMemory(BaseModel):
    """Agent memory structure"""
    experiences: List[Dict[str, Any]] = []
    learned_patterns: List[str] = []
    successful_strategies: List[str] = []
    failed_strategies: List[str] = []

class CoreAgent:
    """
    Core agent class that represents an AI agent stored as a Neo4j node
    with capabilities for learning, evolution, and self-organization.
    """
    
    def __init__(self, 
                 agent_id: str = None,
                 name: str = "",
                 agent_type: str = "GenericAgent",
                 tier: int = 5,
                 instructions: str = "",
                 capabilities: List[str] = None,
                 parent_id: str = None):
        
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.agent_type = agent_type
        self.tier = tier
        self.instructions = instructions
        self.capabilities = capabilities or []
        self.parent_id = parent_id
        self.status = AgentStatus.ACTIVE
        self.performance_metrics = PerformanceMetrics()
        self.memory = AgentMemory()
        self.tools: List[Callable] = []
        
        # Create or update agent node in Neo4j
        self._ensure_agent_exists()
    
    def _ensure_agent_exists(self):
        """Create or update the agent node in Neo4j"""
        with driver.session() as session:
            session.run(
                """
                MERGE (a:Agent {id: $agent_id})
                SET a.name = $name,
                    a.agent_type = $agent_type,
                    a.tier = $tier,
                    a.instructions = $instructions,
                    a.capabilities = $capabilities,
                    a.status = $status,
                    a.created_at = CASE WHEN a.created_at IS NULL THEN datetime() ELSE a.created_at END,
                    a.updated_at = datetime()
                WITH a
                // Create parent relationship if specified
                OPTIONAL MATCH (parent:Agent {id: $parent_id})
                FOREACH (p IN CASE WHEN parent IS NOT NULL THEN [parent] ELSE [] END |
                    MERGE (a)-[:REPORTS_TO]->(p)
                )
                RETURN a.id
                """,
                agent_id=self.agent_id,
                name=self.name,
                agent_type=self.agent_type,
                tier=self.tier,
                instructions=self.instructions,
                capabilities=self.capabilities,
                status=self.status.value,
                parent_id=self.parent_id
            )
    
    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Execute a task and record the outcome in memory
        """
        task_id = str(uuid.uuid4())
        logger.info(f"Agent {self.name} executing task {task_id}")
        
        try:
            # This is a template method - subclasses will implement actual task logic
            result = await self._perform_task(task)
            
            # Calculate score based on result
            score = self._evaluate_performance(task, result)
            outcome = TaskOutcome.SUCCESS if score > 0.7 else TaskOutcome.PARTIAL if score > 0.3 else TaskOutcome.FAILURE
            
            # Create task result
            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                outcome=outcome,
                score=score,
                details=result
            )
            
            # Record in memory and update performance
            await self._record_task_outcome(task_result)
            
            # Check if evolution is needed
            await self._check_evolution_trigger()
            
            return task_result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                outcome=TaskOutcome.FAILURE,
                score=0.0,
                details={"error": str(e)}
            )
    
    async def _perform_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Template method for task execution - to be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _perform_task")
    
    def _evaluate_performance(self, task: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Evaluate task performance - to be customized by subclasses
        """
        # Default simple evaluation
        if "error" in result:
            return 0.0
        return 1.0 if result.get("success", False) else 0.5
    
    async def _record_task_outcome(self, task_result: TaskResult):
        """Record task outcome in Neo4j and update performance metrics"""
        with driver.session() as session:
            # Create task outcome node and link to agent
            session.run(
                """
                MATCH (a:Agent {id: $agent_id})
                CREATE (t:TaskOutcome {
                    id: $task_id,
                    outcome: $outcome,
                    score: $score,
                    details: $details,
                    timestamp: $timestamp
                })
                CREATE (a)-[:PERFORMED]->(t)
                """,
                agent_id=self.agent_id,
                task_id=task_result.task_id,
                outcome=task_result.outcome.value,
                score=task_result.score,
                details=json.dumps(task_result.details),
                timestamp=task_result.timestamp.isoformat()
            )
        
        # Update performance metrics
        self.performance_metrics.total_tasks += 1
        self.performance_metrics.recent_performance.append(task_result.score)
        
        # Keep only last 20 scores for recent performance
        if len(self.performance_metrics.recent_performance) > 20:
            self.performance_metrics.recent_performance = self.performance_metrics.recent_performance[-20:]
        
        # Calculate running averages
        total_score = sum(self.performance_metrics.recent_performance)
        self.performance_metrics.average_score = total_score / len(self.performance_metrics.recent_performance)
        self.performance_metrics.success_rate = len([s for s in self.performance_metrics.recent_performance if s > 0.7]) / len(self.performance_metrics.recent_performance)
        
        # Update in Neo4j
        await self._update_performance_in_db()
    
    async def _update_performance_in_db(self):
        """Update performance metrics in Neo4j"""
        with driver.session() as session:
            session.run(
                """
                MATCH (a:Agent {id: $agent_id})
                SET a.success_rate = $success_rate,
                    a.average_score = $average_score,
                    a.total_tasks = $total_tasks,
                    a.updated_at = datetime()
                """,
                agent_id=self.agent_id,
                success_rate=self.performance_metrics.success_rate,
                average_score=self.performance_metrics.average_score,
                total_tasks=self.performance_metrics.total_tasks
            )
    
    async def _check_evolution_trigger(self):
        """Check if agent should evolve based on performance"""
        # Trigger evolution every 10 tasks or if performance is consistently low
        should_evolve = (
            self.performance_metrics.total_tasks > 0 and 
            self.performance_metrics.total_tasks % 10 == 0
        ) or (
            len(self.performance_metrics.recent_performance) >= 5 and
            self.performance_metrics.average_score < 0.4
        )
        
        if should_evolve:
            await self.evolve()
    
    async def evolve(self):
        """
        Evolve the agent by analyzing performance and modifying instructions
        """
        logger.info(f"Agent {self.name} starting evolution process")
        self.status = AgentStatus.EVOLVING
        
        try:
            # Analyze performance patterns
            analysis = await self._analyze_performance_patterns()
            
            # Generate new instructions based on analysis
            new_instructions = await self._generate_evolved_instructions(analysis)
            
            # Update instructions if they're different
            if new_instructions != self.instructions:
                logger.info(f"Agent {self.name} evolved instructions")
                self.instructions = new_instructions
                await self._update_instructions_in_db()
                
            self.performance_metrics.last_evolution = datetime.now(timezone.utc)
            self.status = AgentStatus.ACTIVE
            
        except Exception as e:
            logger.error(f"Evolution failed for agent {self.name}: {e}")
            self.status = AgentStatus.ACTIVE
    
    async def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze recent performance to identify patterns"""
        with driver.session() as session:
            result = session.run(
                """
                MATCH (a:Agent {id: $agent_id})-[:PERFORMED]->(t:TaskOutcome)
                WHERE t.timestamp > datetime() - duration('P7D')  // Last 7 days
                RETURN t.outcome as outcome, t.score as score, t.details as details, t.timestamp as timestamp
                ORDER BY t.timestamp DESC
                LIMIT 50
                """,
                agent_id=self.agent_id
            )
            
            recent_tasks = []
            for record in result:
                try:
                    details = json.loads(record["details"]) if record["details"] else {}
                except:
                    details = {}
                
                recent_tasks.append({
                    "outcome": record["outcome"],
                    "score": record["score"],
                    "details": details,
                    "timestamp": record["timestamp"]
                })
        
        # Simple pattern analysis
        success_tasks = [t for t in recent_tasks if t["score"] > 0.7]
        failed_tasks = [t for t in recent_tasks if t["score"] < 0.3]
        
        return {
            "total_recent_tasks": len(recent_tasks),
            "success_count": len(success_tasks),
            "failure_count": len(failed_tasks),
            "average_score": sum(t["score"] for t in recent_tasks) / len(recent_tasks) if recent_tasks else 0,
            "successful_strategies": self._extract_patterns(success_tasks),
            "failed_strategies": self._extract_patterns(failed_tasks)
        }
    
    def _extract_patterns(self, tasks: List[Dict]) -> List[str]:
        """Extract common patterns from tasks"""
        # Simple pattern extraction - can be enhanced with ML
        patterns = []
        for task in tasks:
            if task["details"]:
                # Look for common keys or values that might indicate strategies
                for key, value in task["details"].items():
                    if isinstance(value, str) and len(value) < 100:
                        patterns.append(f"{key}:{value}")
        
        # Return most common patterns
        from collections import Counter
        return [pattern for pattern, count in Counter(patterns).most_common(5)]
    
    async def _generate_evolved_instructions(self, analysis: Dict[str, Any]) -> str:
        """Generate evolved instructions based on performance analysis"""
        # Simple instruction evolution - enhance with LLM in production
        current_instructions = self.instructions
        
        if analysis["average_score"] < 0.5:
            # Performance is poor, try to incorporate successful strategies
            successful_strategies = analysis.get("successful_strategies", [])
            if successful_strategies:
                addition = f"\nImproved Strategy: Focus on {', '.join(successful_strategies[:2])}"
                return current_instructions + addition
        
        elif analysis["average_score"] > 0.8:
            # Performance is good, document the working approach
            addition = f"\nProven Approach: Current strategy is working well with {analysis['success_count']} recent successes"
            return current_instructions + addition
        
        return current_instructions
    
    async def _update_instructions_in_db(self):
        """Update instructions in Neo4j"""
        with driver.session() as session:
            session.run(
                """
                MATCH (a:Agent {id: $agent_id})
                SET a.instructions = $instructions,
                    a.last_evolution = datetime(),
                    a.updated_at = datetime()
                """,
                agent_id=self.agent_id,
                instructions=self.instructions
            )
    
    def add_tool(self, tool: Callable):
        """Add a tool function to this agent"""
        self.tools.append(tool)
    
    @classmethod
    async def load_agent(cls, agent_id: str) -> Optional['CoreAgent']:
        """Load an existing agent from Neo4j"""
        with driver.session() as session:
            result = session.run(
                """
                MATCH (a:Agent {id: $agent_id})
                OPTIONAL MATCH (a)-[:REPORTS_TO]->(parent:Agent)
                RETURN a, parent.id as parent_id
                """,
                agent_id=agent_id
            )
            
            record = result.single()
            if record:
                agent_data = record["a"]
                agent = cls(
                    agent_id=agent_data["id"],
                    name=agent_data["name"],
                    agent_type=agent_data["agent_type"],
                    tier=agent_data["tier"],
                    instructions=agent_data["instructions"],
                    capabilities=agent_data.get("capabilities", []),
                    parent_id=record["parent_id"]
                )
                
                # Load performance metrics
                agent.performance_metrics.success_rate = agent_data.get("success_rate", 0.0)
                agent.performance_metrics.average_score = agent_data.get("average_score", 0.0)
                agent.performance_metrics.total_tasks = agent_data.get("total_tasks", 0)
                
                return agent
        
        return None
    
    async def get_subordinates(self) -> List['CoreAgent']:
        """Get all agents that report to this agent"""
        with driver.session() as session:
            result = session.run(
                """
                MATCH (subordinate:Agent)-[:REPORTS_TO]->(a:Agent {id: $agent_id})
                RETURN subordinate.id as subordinate_id
                """,
                agent_id=self.agent_id
            )
            
            subordinates = []
            for record in result:
                subordinate = await self.load_agent(record["subordinate_id"])
                if subordinate:
                    subordinates.append(subordinate)
            
            return subordinates
    
    def __repr__(self):
        return f"CoreAgent(id={self.agent_id}, name={self.name}, type={self.agent_type}, tier={self.tier})"


class AgentOrchestrator:
    """
    Central orchestrator for managing the agent network
    """
    
    def __init__(self):
        self.active_agents: Dict[str, CoreAgent] = {}
    
    async def get_agent(self, agent_id: str) -> Optional[CoreAgent]:
        """Get agent by ID, loading from database if necessary"""
        if agent_id not in self.active_agents:
            agent = await CoreAgent.load_agent(agent_id)
            if agent:
                self.active_agents[agent_id] = agent
        
        return self.active_agents.get(agent_id)
    
    async def get_agents_by_type(self, agent_type: str) -> List[CoreAgent]:
        """Get all agents of a specific type"""
        with driver.session() as session:
            result = session.run(
                """
                MATCH (a:Agent {agent_type: $agent_type})
                WHERE a.status = 'active'
                RETURN a.id as agent_id
                """,
                agent_type=agent_type
            )
            
            agents = []
            for record in result:
                agent = await self.get_agent(record["agent_id"])
                if agent:
                    agents.append(agent)
            
            return agents
    
    async def create_agent_hierarchy(self):
        """Create the initial agent hierarchy as specified in the requirements"""
        
        # Create CEO Agent (Tier 1)
        ceo = CoreAgent(
            name="CEO Agent",
            agent_type="CEO_Agent",
            tier=1,
            instructions="""You are the Chief Executive Officer of the AI organization. 
            Your role is strategic orchestration, coordinating between different departments,
            and ensuring overall system efficiency. You make high-level decisions about 
            resource allocation and organizational structure.""",
            capabilities=["strategic_planning", "resource_allocation", "coordination"]
        )
        
        # Create Tier 2 Directors
        cdo = CoreAgent(
            name="Chief Data Officer Agent",
            agent_type="Chief_Data_Officer_Agent", 
            tier=2,
            parent_id=ceo.agent_id,
            instructions="""You are the Chief Data Officer responsible for all data operations.
            Oversee data quality, integration, and ensure data integrity across the system.
            Coordinate with your managers to optimize data processing workflows.""",
            capabilities=["data_governance", "quality_assurance", "data_strategy"]
        )
        
        cao = CoreAgent(
            name="Chief Analytics Officer Agent",
            agent_type="Chief_Analytics_Officer_Agent",
            tier=2,
            parent_id=ceo.agent_id,
            instructions="""You are the Chief Analytics Officer responsible for intelligence generation.
            Oversee pattern detection, insight generation, and analytical processes.
            Transform raw data into actionable intelligence.""",
            capabilities=["analytics", "pattern_recognition", "insight_generation"]
        )
        
        hr_director = CoreAgent(
            name="HR Director Agent",
            agent_type="HR_Director_Agent",
            tier=2,
            parent_id=ceo.agent_id,
            instructions="""You are the HR Director responsible for organizational evolution.
            Create new agents when needed, evaluate agent performance, and coordinate training.
            Ensure the organization adapts and evolves optimally.""",
            capabilities=["agent_creation", "performance_evaluation", "organizational_development"]
        )
        
        qa_director = CoreAgent(
            name="Quality Assurance Director Agent",
            agent_type="Quality_Assurance_Director_Agent",
            tier=2,
            parent_id=ceo.agent_id,
            instructions="""You are the QA Director responsible for system integrity.
            Oversee code quality, output validation, and compliance across all operations.
            Ensure high standards are maintained throughout the organization.""",
            capabilities=["quality_control", "compliance", "validation"]
        )
        
        self.active_agents.update({
            ceo.agent_id: ceo,
            cdo.agent_id: cdo,
            cao.agent_id: cao,
            hr_director.agent_id: hr_director,
            qa_director.agent_id: qa_director
        })
        
        logger.info("Created initial agent hierarchy")
        return ceo, cdo, cao, hr_director, qa_director

# Global orchestrator instance
orchestrator = AgentOrchestrator()
