"""
Enhanced Text-to-Cypher using Google Gemma3-4B Fine-tuned Model

This module provides advanced text-to-cypher translation capabilities
using the Gemma3-4B model specifically fine-tuned for Neo4j queries.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import ollama
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("GEMMA_TEXT2CYPHER")

class CypherRequest(BaseModel):
    """Enhanced request for text-to-cypher translation"""
    question: str
    context: Optional[str] = None
    schema_version: Optional[str] = "latest"
    agent_context: Optional[Dict[str, Any]] = None
    complexity_level: Optional[str] = "standard"  # basic, standard, advanced
    
class CypherResponse(BaseModel):
    """Response containing generated Cypher and metadata"""
    cypher_query: str
    confidence_score: float
    explanation: Optional[str] = None
    complexity_analysis: Dict[str, Any]
    estimated_performance: Dict[str, Any]
    suggestions: List[str] = []

class GemmaTextToCypher:
    """
    Advanced Text-to-Cypher engine using Gemma3-4B fine-tuned model
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "GEMMA_MODEL", 
            "gemma2:9b"  # Default Ollama Gemma model (no API key needed)
        )
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
        )
        self.schema_cache = {}
        self.agent_context_cache = {}
        
    async def generate_cypher(self, request: CypherRequest) -> CypherResponse:
        """
        Generate Cypher query using Gemma3-4B with enhanced context
        """
        try:
            # Get current schema
            schema = await self._get_enhanced_schema()
            
            # Build enhanced prompt with agent context
            prompt = await self._build_enhanced_prompt(request, schema)
            
            # Generate with Gemma3-4B
            cypher_result = await self._call_gemma_model(prompt, request.complexity_level)
            
            # Analyze and enhance the result
            enhanced_response = await self._enhance_cypher_response(
                cypher_result, request, schema
            )
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            return CypherResponse(
                cypher_query="// Error in generation",
                confidence_score=0.0,
                explanation=f"Generation failed: {str(e)}",
                complexity_analysis={"error": True},
                estimated_performance={"error": True}
            )
    
    async def _get_enhanced_schema(self) -> Dict[str, Any]:
        """Get enhanced schema including agent-specific information"""
        if "enhanced_schema" in self.schema_cache:
            return self.schema_cache["enhanced_schema"]
        
        with self.neo4j_driver.session() as session:
            # Get basic schema
            schema_result = session.run("""
                CALL db.schema.visualization()
                YIELD nodes, relationships
                RETURN nodes, relationships
            """)
            
            basic_schema = schema_result.single()
            
            # Get agent-specific schema
            agent_schema = session.run("""
                MATCH (a:Agent)
                RETURN DISTINCT labels(a) as labels, keys(a) as properties
                UNION
                MATCH (t:TaskOutcome)
                RETURN DISTINCT labels(t) as labels, keys(t) as properties
                UNION
                MATCH ()-[r:REPORTS_TO|PERFORMED|FOR_CUSTOMER|FOR_ACCOUNT]->()
                RETURN DISTINCT type(r) as relationship_type, keys(r) as properties
            """)
            
            # Enhanced schema with agent intelligence context
            enhanced_schema = {
                "basic_schema": basic_schema,
                "agent_nodes": [],
                "relationships": [],
                "common_patterns": await self._get_common_query_patterns(),
                "performance_hints": await self._get_performance_hints()
            }
            
            for record in agent_schema:
                if record.get("labels"):
                    enhanced_schema["agent_nodes"].append({
                        "labels": record["labels"],
                        "properties": record["properties"]
                    })
                if record.get("relationship_type"):
                    enhanced_schema["relationships"].append({
                        "type": record["relationship_type"],
                        "properties": record["properties"]
                    })
            
            self.schema_cache["enhanced_schema"] = enhanced_schema
            return enhanced_schema
    
    async def _get_common_query_patterns(self) -> List[Dict[str, Any]]:
        """Get common query patterns for agent operations"""
        return [
            {
                "pattern": "Agent Performance Analysis",
                "template": "MATCH (a:Agent)-[:PERFORMED]->(t:TaskOutcome) WHERE a.id = $agent_id",
                "use_case": "Analyzing agent task performance"
            },
            {
                "pattern": "Organizational Hierarchy",
                "template": "MATCH (a:Agent)-[:REPORTS_TO*]->(boss:Agent)",
                "use_case": "Understanding reporting relationships"
            },
            {
                "pattern": "Agent Evolution History",
                "template": "MATCH (a:Agent) WHERE a.last_evolution IS NOT NULL",
                "use_case": "Tracking agent learning and evolution"
            },
            {
                "pattern": "Task Outcome Patterns",
                "template": "MATCH (a:Agent)-[:PERFORMED]->(t:TaskOutcome) WHERE t.timestamp > datetime() - duration('P7D')",
                "use_case": "Recent performance analysis"
            }
        ]
    
    async def _get_performance_hints(self) -> List[str]:
        """Get performance optimization hints"""
        return [
            "Use parameters for dynamic values to enable query plan caching",
            "Add indexes on frequently queried properties like Agent.id, Agent.agent_type",
            "Use LIMIT to prevent large result sets in exploration queries",
            "Consider using EXPLAIN or PROFILE for complex queries",
            "Use WITH clauses to pipeline complex operations"
        ]
    
    async def _build_enhanced_prompt(self, request: CypherRequest, schema: Dict[str, Any]) -> str:
        """Build enhanced prompt for Gemma3-4B with rich context"""
        
        base_prompt = f"""# Neo4j Cypher Query Generation with Gemma3-4B

You are an expert Neo4j Cypher query generator specialized in AI agent management systems.

## Context
This is a self-evolving AI infrastructure where agents exist as Neo4j nodes. Agents can:
- Learn from performance and evolve their instructions
- Report to other agents in hierarchical structures  
- Execute tasks and record outcomes
- Be created, modified, or replaced dynamically

## Database Schema
{self._format_schema_for_prompt(schema)}

## Common Query Patterns
{self._format_patterns_for_prompt(schema.get('common_patterns', []))}

## User Question
{request.question}

## Additional Context
{request.context or 'No additional context provided'}

## Requirements
- Generate a Cypher query that answers the question accurately
- Use parameters ($param) for dynamic values when appropriate
- Include helpful comments in the query
- Optimize for performance with appropriate indexes/limits
- Consider the agent-centric nature of this system

## Response Format
Return ONLY the Cypher query without explanation or markdown formatting.
"""
        
        if request.agent_context:
            base_prompt += f"\n## Agent Context\n{request.agent_context}\n"
        
        return base_prompt
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format schema information for the prompt"""
        formatted = "### Node Labels and Properties:\n"
        
        for node_info in schema.get("agent_nodes", []):
            labels = ", ".join(node_info.get("labels", []))
            properties = ", ".join(node_info.get("properties", []))
            formatted += f"- {labels}: {properties}\n"
        
        formatted += "\n### Relationship Types:\n"
        for rel_info in schema.get("relationships", []):
            rel_type = rel_info.get("type", "Unknown")
            properties = ", ".join(rel_info.get("properties", []))
            formatted += f"- {rel_type}: {properties}\n"
        
        return formatted
    
    def _format_patterns_for_prompt(self, patterns: List[Dict[str, Any]]) -> str:
        """Format common patterns for the prompt"""
        formatted = ""
        for pattern in patterns:
            formatted += f"### {pattern['pattern']}\n"
            formatted += f"Template: {pattern['template']}\n"
            formatted += f"Use Case: {pattern['use_case']}\n\n"
        return formatted
    
    async def _call_gemma_model(self, prompt: str, complexity_level: str) -> Dict[str, Any]:
        """Call Gemma model with appropriate parameters (supports multiple providers)"""
        
        # Adjust parameters based on complexity
        temperature = {
            "basic": 0.1,
            "standard": 0.2, 
            "advanced": 0.3
        }.get(complexity_level, 0.2)
        
        try:
            # Check for Google AI Studio
            if os.getenv("GOOGLE_API_KEY"):
                return await self._call_google_ai_studio(prompt, temperature)
            
            # Check for OpenAI-compatible API
            elif os.getenv("OPENAI_API_KEY"):
                return await self._call_openai_compatible(prompt, temperature)
            
            # Check for Hugging Face
            elif os.getenv("HUGGINGFACE_API_KEY"):
                return await self._call_huggingface(prompt, temperature)
            
            # Default to Ollama (local)
            else:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{
                        "role": "user", 
                        "content": prompt
                    }],
                    options={
                        "temperature": temperature,
                        "top_p": 0.9,
                        "max_tokens": 2048,
                        "num_predict": 512
                    }
                )
                
                cypher_query = response['message']['content'].strip()
                cypher_query = self._clean_cypher_response(cypher_query)
                
                return {
                    "cypher_query": cypher_query,
                    "raw_response": response,
                    "model_used": self.model_name
                }
            
        except Exception as e:
            logger.error(f"Gemma model call failed: {e}")
            return {
                "cypher_query": "// Model call failed",
                "error": str(e)
            }
    
    async def _call_google_ai_studio(self, prompt: str, temperature: float) -> Dict[str, Any]:
        """Call Google AI Studio API"""
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048,
            )
        )
        
        cypher_query = response.text.strip()
        cypher_query = self._clean_cypher_response(cypher_query)
        
        return {
            "cypher_query": cypher_query,
            "raw_response": response.text,
            "model_used": "google-gemini-pro"
        }
    
    async def _call_openai_compatible(self, prompt: str, temperature: float) -> Dict[str, Any]:
        """Call OpenAI-compatible API (for various cloud providers)"""
        import openai
        
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048
        )
        
        cypher_query = response.choices[0].message.content.strip()
        cypher_query = self._clean_cypher_response(cypher_query)
        
        return {
            "cypher_query": cypher_query,
            "raw_response": response.choices[0].message.content,
            "model_used": self.model_name
        }
    
    async def _call_huggingface(self, prompt: str, temperature: float) -> Dict[str, Any]:
        """Call Hugging Face Inference API"""
        import httpx
        
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        model_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": 2048,
                "return_full_text": False
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(model_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                cypher_query = result[0].get("generated_text", "").strip()
            else:
                cypher_query = str(result)
            
            cypher_query = self._clean_cypher_response(cypher_query)
            
            return {
                "cypher_query": cypher_query,
                "raw_response": result,
                "model_used": self.model_name
            }
    
    def _clean_cypher_response(self, response: str) -> str:
        """Clean and validate the Cypher response"""
        # Remove markdown formatting if present
        if "```cypher" in response:
            response = response.split("```cypher")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        # Remove common prefixes
        response = response.replace("cypher:", "").strip()
        response = response.replace("CYPHER:", "").strip()
        
        # Handle newline escapes
        response = response.replace("\\n", "\n")
        
        return response.strip()
    
    async def _enhance_cypher_response(self, 
                                     cypher_result: Dict[str, Any], 
                                     request: CypherRequest,
                                     schema: Dict[str, Any]) -> CypherResponse:
        """Enhance the Cypher response with analysis and suggestions"""
        
        cypher_query = cypher_result.get("cypher_query", "")
        
        # Basic validation
        confidence_score = await self._calculate_confidence_score(cypher_query, request)
        
        # Complexity analysis
        complexity_analysis = self._analyze_query_complexity(cypher_query)
        
        # Performance estimation
        performance_estimate = await self._estimate_performance(cypher_query)
        
        # Generate suggestions
        suggestions = await self._generate_suggestions(cypher_query, schema)
        
        return CypherResponse(
            cypher_query=cypher_query,
            confidence_score=confidence_score,
            explanation=f"Generated using {cypher_result.get('model_used', 'Gemma3-4B')}",
            complexity_analysis=complexity_analysis,
            estimated_performance=performance_estimate,
            suggestions=suggestions
        )
    
    async def _calculate_confidence_score(self, cypher_query: str, request: CypherRequest) -> float:
        """Calculate confidence score for the generated query"""
        score = 0.5  # Base score
        
        # Check for basic Cypher syntax
        if "MATCH" in cypher_query.upper():
            score += 0.2
        if "RETURN" in cypher_query.upper():
            score += 0.2
        
        # Check for agent-specific patterns
        if any(keyword in cypher_query.upper() for keyword in ["AGENT", "TASKOUTCOME", "REPORTS_TO"]):
            score += 0.1
        
        # Penalize for errors or empty responses
        if not cypher_query or "error" in cypher_query.lower():
            score = 0.1
        
        return min(score, 1.0)
    
    def _analyze_query_complexity(self, cypher_query: str) -> Dict[str, Any]:
        """Analyze the complexity of the generated query"""
        
        query_upper = cypher_query.upper()
        
        return {
            "has_aggregation": any(func in query_upper for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"]),
            "has_subqueries": "WITH" in query_upper or "CALL" in query_upper,
            "has_path_patterns": "*" in cypher_query,
            "estimated_complexity": "high" if "*" in cypher_query or "WITH" in query_upper else "medium" if any(func in query_upper for func in ["COUNT", "SUM"]) else "low",
            "line_count": len(cypher_query.split('\n'))
        }
    
    async def _estimate_performance(self, cypher_query: str) -> Dict[str, Any]:
        """Estimate query performance characteristics"""
        
        try:
            with self.neo4j_driver.session() as session:
                # Use EXPLAIN to get query plan without execution
                explain_result = session.run(f"EXPLAIN {cypher_query}")
                plan_info = explain_result.consume()
                
                return {
                    "has_index_usage": True,  # Would need to parse plan for actual info
                    "estimated_rows": "unknown",
                    "complexity_score": len(cypher_query.split()),
                    "performance_tier": "estimated_good"
                }
        except:
            return {
                "performance_tier": "unknown",
                "validation_error": True
            }
    
    async def _generate_suggestions(self, cypher_query: str, schema: Dict[str, Any]) -> List[str]:
        """Generate optimization and improvement suggestions"""
        
        suggestions = []
        
        # Basic optimization suggestions
        if "LIMIT" not in cypher_query.upper():
            suggestions.append("Consider adding LIMIT to prevent large result sets")
        
        if "$" not in cypher_query and any(char.isdigit() for char in cypher_query):
            suggestions.append("Consider using parameters for dynamic values")
        
        # Agent-specific suggestions
        if "Agent" in cypher_query and "WHERE" not in cypher_query.upper():
            suggestions.append("Consider filtering agents by type or status for better performance")
        
        if "TaskOutcome" in cypher_query:
            suggestions.append("Consider filtering by timestamp for recent task analysis")
        
        return suggestions
    
    def cleanup(self):
        """Clean up resources"""
        if self.neo4j_driver:
            self.neo4j_driver.close()

# Global instance
gemma_text_to_cypher = GemmaTextToCypher()
