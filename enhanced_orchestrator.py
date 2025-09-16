"""
Enhanced Main Orchestrator with Gemma3-4B Text-to-Cypher and Data-Driven Agents

This orchestrator demonstrates the evolution in action with:
1. Gemma3-4B powered natural language to Cypher queries
2. Fully data-driven agents stored in Neo4j
3. Real-time agent modification through database operations
4. Advanced evolution capabilities
"""

import asyncio
import logging
import argparse
import json
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
from core_agent import driver
from gemma_text_to_cypher import gemma_text_to_cypher, CypherRequest
from data_driven_agents import data_driven_agent_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_agentic_intelligence.log')
    ]
)
logger = logging.getLogger("ENHANCED_ORCHESTRATOR")

class EnhancedOrchestrator:
    """
    Enhanced orchestrator with data-driven agents and Gemma3-4B integration
    """
    
    def __init__(self):
        self.running = True
        self.current_task = None
        
    async def initialize_enhanced_system(self):
        """Initialize the enhanced system with data-driven architecture"""
        logger.info("Initializing Enhanced Self-Evolving AI Infrastructure")
        
        try:
            # Setup Neo4j schema for enhanced features
            await self._setup_enhanced_schema()
            
            # Initialize agent templates in database
            await data_driven_agent_manager.initialize_agent_templates()
            
            # Create initial agent hierarchy using templates
            await self._create_data_driven_hierarchy()
            
            logger.info("Enhanced system initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced system initialization failed: {e}")
            return False
    
    async def _setup_enhanced_schema(self):
        """Setup enhanced Neo4j schema"""
        with driver.session() as session:
            schema_queries = [
                # Core agent constraints
                "CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
                "CREATE INDEX agent_type_index IF NOT EXISTS FOR (a:Agent) ON (a.agent_type)",
                "CREATE INDEX agent_status_index IF NOT EXISTS FOR (a:Agent) ON (a.status)",
                "CREATE INDEX agent_tier_index IF NOT EXISTS FOR (a:Agent) ON (a.tier)",
                
                # Agent template constraints
                "CREATE CONSTRAINT template_id_unique IF NOT EXISTS FOR (t:AgentTemplate) REQUIRE t.template_id IS UNIQUE",
                "CREATE INDEX template_type_index IF NOT EXISTS FOR (t:AgentTemplate) ON (t.agent_type)",
                
                # Task outcome indexes
                "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:TaskOutcome) REQUIRE t.id IS UNIQUE",
                "CREATE INDEX task_timestamp_index IF NOT EXISTS FOR (t:TaskOutcome) ON (t.timestamp)",
                "CREATE INDEX task_outcome_index IF NOT EXISTS FOR (t:TaskOutcome) ON (t.outcome)",
                
                # Performance indexes
                "CREATE INDEX agent_performance_index IF NOT EXISTS FOR (a:Agent) ON (a.success_rate, a.average_score)",
                
                # Evolution tracking indexes
                "CREATE INDEX agent_evolution_index IF NOT EXISTS FOR (a:Agent) ON (a.last_evolution)",
                "CREATE INDEX agent_creation_index IF NOT EXISTS FOR (a:Agent) ON (a.created_at)"
            ]
            
            for query in schema_queries:
                try:
                    session.run(query)
                    logger.debug(f"Executed enhanced schema query: {query}")
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {e}")
    
    async def _create_data_driven_hierarchy(self):
        """Create agent hierarchy using data-driven templates"""
        
        # Create CEO
        ceo_id = await data_driven_agent_manager.create_agent_from_template(
            template_id="ceo_agent_template",
            custom_name="Strategic CEO Agent"
        )
        
        # Create HR Director reporting to CEO
        hr_director_id = await data_driven_agent_manager.create_agent_from_template(
            template_id="hr_director_template",
            custom_name="Organizational Evolution Director",
            parent_id=ceo_id
        )
        
        # Create Entity Resolution Agents with different strategies
        conservative_id = await data_driven_agent_manager.create_agent_from_template(
            template_id="entity_resolution_conservative_template",
            custom_name="Precision-Focused Matcher",
            parent_id=hr_director_id
        )
        
        aggressive_id = await data_driven_agent_manager.create_agent_from_template(
            template_id="entity_resolution_aggressive_template", 
            custom_name="Recall-Optimized Matcher",
            parent_id=hr_director_id
        )
        
        balanced_id = await data_driven_agent_manager.create_agent_from_template(
            template_id="entity_resolution_balanced_template",
            custom_name="F1-Balanced Matcher",
            parent_id=hr_director_id
        )
        
        logger.info(f"Created data-driven hierarchy: CEO -> HR Director -> 3 Entity Resolution Agents")
        return {
            "ceo_id": ceo_id,
            "hr_director_id": hr_director_id,
            "entity_agents": [conservative_id, aggressive_id, balanced_id]
        }
    
    async def run_enhanced_interactive_mode(self):
        """Enhanced interactive mode with Gemma3-4B and data-driven features"""
        logger.info("Starting Enhanced Interactive Mode")
        
        print("\n" + "="*80)
        print("🧠 ENHANCED SELF-EVOLVING AI INFRASTRUCTURE")
        print("="*80)
        print("New Capabilities:")
        print("• 🤖 Gemma3-4B powered natural language to Cypher queries")
        print("• 📊 Fully data-driven agents stored in Neo4j")
        print("• 🔄 Real-time agent modification through database")
        print("• 🧬 Advanced evolution with performance analysis")
        print("\nCommands:")
        print("  'ask <question>' - Natural language query with Gemma3-4B")
        print("  'agents' - Show all data-driven agents")
        print("  'modify <agent_id>' - Modify agent through database")
        print("  'evolve <agent_id>' - Trigger agent evolution")
        print("  'templates' - Show available agent templates")
        print("  'create <template_id>' - Create new agent from template")
        print("  'demo' - Run evolution demonstration")
        print("  'status' - System status")
        print("  'quit' - Exit")
        print("="*80)
        
        while self.running:
            try:
                command = input("\n🤖 Enter command: ").strip()
                
                if command.lower() in ['quit', 'exit']:
                    self.running = False
                    break
                elif command.startswith('ask '):
                    await self._handle_natural_language_query(command[4:])
                elif command == 'agents':
                    await self._handle_data_driven_agents_command()
                elif command.startswith('modify '):
                    await self._handle_modify_agent_command(command[7:])
                elif command.startswith('evolve '):
                    await self._handle_evolve_agent_command(command[7:])
                elif command == 'templates':
                    await self._handle_templates_command()
                elif command.startswith('create '):
                    await self._handle_create_agent_command(command[7:])
                elif command == 'demo':
                    await self._handle_evolution_demo()
                elif command == 'status':
                    await self._handle_enhanced_status_command()
                elif command == 'help':
                    print("Use 'ask <question>' for natural language queries, or see commands above.")
                else:
                    print(f"Unknown command: '{command}'. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nReceived interrupt signal. Shutting down gracefully...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in enhanced interactive mode: {e}")
                print(f"Error: {e}")
        
        print("\n🎯 Enhanced system shutting down...")
    
    async def _handle_natural_language_query(self, question: str):
        """Handle natural language queries using Gemma3-4B"""
        print(f"\n🔍 Processing: '{question}'")
        print("🤖 Using Gemma3-4B for Text-to-Cypher translation...")
        
        try:
            # Create enhanced request
            request = CypherRequest(
                question=question,
                context="Self-evolving AI infrastructure with agents as Neo4j nodes",
                agent_context={
                    "system_type": "agentic_intelligence",
                    "focus": "agent_performance_and_evolution"
                }
            )
            
            # Generate Cypher with Gemma3-4B
            response = await gemma_text_to_cypher.generate_cypher(request)
            
            print(f"\n📝 Generated Cypher Query:")
            print(f"```cypher\n{response.cypher_query}\n```")
            print(f"🎯 Confidence Score: {response.confidence_score:.2%}")
            
            if response.suggestions:
                print(f"\n💡 Suggestions:")
                for suggestion in response.suggestions:
                    print(f"  • {suggestion}")
            
            # Execute the query if confidence is high enough
            if response.confidence_score > 0.6:
                print(f"\n⚡ Executing query...")
                await self._execute_cypher_query(response.cypher_query)
            else:
                print(f"\n⚠️ Confidence too low for automatic execution. Review the query manually.")
                
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
    
    async def _execute_cypher_query(self, cypher_query: str):
        """Execute Cypher query and display results"""
        try:
            with driver.session() as session:
                result = session.run(cypher_query)
                records = list(result)
                
                if records:
                    print(f"\n📊 Results ({len(records)} rows):")
                    print("-" * 60)
                    
                    # Display first few records
                    for i, record in enumerate(records[:10]):
                        print(f"Row {i+1}: {dict(record)}")
                    
                    if len(records) > 10:
                        print(f"... and {len(records) - 10} more rows")
                else:
                    print("\n📊 No results returned")
                    
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
    
    async def _handle_data_driven_agents_command(self):
        """Show all data-driven agents"""
        print("\n📋 Data-Driven Agents:")
        print("-" * 70)
        
        try:
            # Get all agents
            all_agents = await data_driven_agent_manager.get_agents_by_criteria()
            
            for agent in all_agents:
                print(f"🤖 {agent.name}")
                print(f"   Type: {agent.agent_type} | Tier: {agent.tier} | Status: {agent.status}")
                print(f"   ID: {agent.agent_id}")
                print(f"   Capabilities: {', '.join(agent.capabilities)}")
                
                # Show parameters
                params = agent.parameters
                if params:
                    key_params = {k: v for k, v in params.items() if k in ['match_threshold', 'decision_threshold']}
                    if key_params:
                        print(f"   Key Parameters: {key_params}")
                print()
                
        except Exception as e:
            print(f"❌ Error retrieving agents: {e}")
    
    async def _handle_modify_agent_command(self, agent_id: str):
        """Modify agent through database operations"""
        if not agent_id:
            print("Usage: modify <agent_id>")
            return
        
        try:
            # Load agent
            agent = await data_driven_agent_manager.load_agent_from_database(agent_id)
            
            print(f"\n🔧 Modifying Agent: {agent.name}")
            print(f"Current parameters: {agent.parameters}")
            
            print("\nWhat would you like to modify?")
            print("1. Instructions")
            print("2. Parameters") 
            print("3. Status")
            print("4. Capabilities")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                new_instructions = input("Enter new instructions: ")
                await data_driven_agent_manager.update_agent_in_database(
                    agent_id, {"instructions": new_instructions}
                )
                print("✅ Instructions updated!")
                
            elif choice == "2":
                param_name = input("Parameter name: ")
                param_value = input("Parameter value: ")
                
                # Try to convert to appropriate type
                try:
                    if param_value.lower() in ['true', 'false']:
                        param_value = param_value.lower() == 'true'
                    elif param_value.replace('.', '').isdigit():
                        param_value = float(param_value) if '.' in param_value else int(param_value)
                except:
                    pass  # Keep as string
                
                current_params = agent.parameters
                current_params[param_name] = param_value
                
                await data_driven_agent_manager.update_agent_in_database(
                    agent_id, {"parameters": json.dumps(current_params)}
                )
                print(f"✅ Parameter {param_name} updated to {param_value}!")
                
            elif choice == "3":
                new_status = input("Enter new status (active/learning/evolving/deprecated): ")
                await data_driven_agent_manager.update_agent_in_database(
                    agent_id, {"status": new_status}
                )
                print(f"✅ Status updated to {new_status}!")
                
            elif choice == "4":
                new_capability = input("Enter new capability to add: ")
                current_capabilities = agent.capabilities
                if new_capability not in current_capabilities:
                    current_capabilities.append(new_capability)
                    await data_driven_agent_manager.update_agent_in_database(
                        agent_id, {"capabilities": current_capabilities}
                    )
                    print(f"✅ Added capability: {new_capability}!")
                else:
                    print("Capability already exists!")
            
        except Exception as e:
            print(f"❌ Modification failed: {e}")
    
    async def _handle_evolve_agent_command(self, agent_id: str):
        """Trigger agent evolution"""
        if not agent_id:
            print("Usage: evolve <agent_id>")
            return
        
        try:
            agent = await data_driven_agent_manager.load_agent_from_database(agent_id)
            
            print(f"\n🧬 Triggering evolution for: {agent.name}")
            print("Analyzing performance patterns...")
            
            await agent.evolve()
            
            print("✅ Agent evolution completed!")
            print("🔄 Instructions have been updated based on performance analysis")
            
        except Exception as e:
            print(f"❌ Evolution failed: {e}")
    
    async def _handle_templates_command(self):
        """Show available agent templates"""
        print("\n📋 Available Agent Templates:")
        print("-" * 50)
        
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (t:AgentTemplate)
                    RETURN t.template_id, t.name, t.agent_type, t.tier
                    ORDER BY t.tier, t.name
                """)
                
                for record in result:
                    print(f"🎯 {record['t.template_id']}")
                    print(f"   Name: {record['t.name']}")
                    print(f"   Type: {record['t.agent_type']} | Tier: {record['t.tier']}")
                    print()
                    
        except Exception as e:
            print(f"❌ Error retrieving templates: {e}")
    
    async def _handle_create_agent_command(self, template_id: str):
        """Create new agent from template"""
        if not template_id:
            print("Usage: create <template_id>")
            return
        
        try:
            custom_name = input("Enter custom name (or press Enter for default): ").strip()
            if not custom_name:
                custom_name = None
            
            agent_id = await data_driven_agent_manager.create_agent_from_template(
                template_id=template_id,
                custom_name=custom_name
            )
            
            print(f"✅ Created new agent!")
            print(f"   Agent ID: {agent_id}")
            print(f"   Template: {template_id}")
            
        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
    
    async def _handle_evolution_demo(self):
        """Run evolution demonstration"""
        print("\n🎬 Evolution Demo: Watch Agents Learn and Adapt")
        print("=" * 60)
        
        try:
            # Get entity resolution agents
            entity_agents = await data_driven_agent_manager.get_agents_by_criteria(
                agent_type="Entity_Resolution_Agent"
            )
            
            if len(entity_agents) < 3:
                print("⚠️ Need at least 3 Entity Resolution Agents. Creating them...")
                hierarchy = await self._create_data_driven_hierarchy()
                entity_agents = await data_driven_agent_manager.get_agents_by_criteria(
                    agent_type="Entity_Resolution_Agent"
                )
            
            print(f"🤖 Found {len(entity_agents)} Entity Resolution Agents")
            
            # Simulate different performance scenarios
            scenarios = [
                {"name": "High Performance", "score": 0.9},
                {"name": "Medium Performance", "score": 0.6}, 
                {"name": "Poor Performance", "score": 0.3}
            ]
            
            for i, agent in enumerate(entity_agents[:3]):
                scenario = scenarios[i]
                print(f"\n🎯 Simulating {scenario['name']} for {agent.name}")
                
                # Simulate task execution
                task_result = await agent.execute_task({
                    "type": "performance_simulation",
                    "simulated_score": scenario["score"]
                })
                
                print(f"   Result: {task_result.outcome.value} (score: {task_result.score:.2f})")
                
                # Trigger evolution for poor performers
                if task_result.score < 0.5:
                    print(f"   🧬 Triggering evolution (performance below threshold)")
                    await agent.evolve()
                    print(f"   ✅ Evolution completed")
            
            print(f"\n🎉 Evolution demo completed! Check Neo4j to see the changes.")
            
        except Exception as e:
            print(f"❌ Evolution demo failed: {e}")
    
    async def _handle_enhanced_status_command(self):
        """Show enhanced system status"""
        print("\n📊 Enhanced System Status:")
        print("-" * 50)
        
        try:
            with driver.session() as session:
                # Agent statistics
                agent_stats = session.run("""
                    MATCH (a:Agent)
                    RETURN 
                        a.agent_type as type,
                        count(*) as count,
                        avg(a.success_rate) as avg_success_rate,
                        avg(a.total_tasks) as avg_tasks
                    ORDER BY type
                """)
                
                print("🤖 Agent Population:")
                total_agents = 0
                for record in agent_stats:
                    agent_type = record["type"]
                    count = record["count"]
                    avg_success = record["avg_success_rate"] or 0
                    avg_tasks = record["avg_tasks"] or 0
                    total_agents += count
                    print(f"  {agent_type}: {count} agents (avg success: {avg_success:.1%}, avg tasks: {avg_tasks:.0f})")
                
                print(f"\nTotal Agents: {total_agents}")
                
                # Template statistics
                template_stats = session.run("""
                    MATCH (t:AgentTemplate)
                    RETURN count(*) as template_count
                """)
                
                template_count = template_stats.single()["template_count"]
                print(f"Available Templates: {template_count}")
                
                # Recent activity
                recent_activity = session.run("""
                    MATCH (a:Agent)-[:PERFORMED]->(t:TaskOutcome)
                    WHERE t.timestamp > datetime() - duration('PT1H')
                    RETURN count(*) as recent_tasks
                """)
                
                recent_count = recent_activity.single()["recent_tasks"]
                print(f"Tasks in Last Hour: {recent_count}")
                
                # Evolution activity
                evolution_stats = session.run("""
                    MATCH (a:Agent)
                    WHERE a.last_evolution IS NOT NULL
                    RETURN count(*) as evolved_agents
                """)
                
                evolved_count = evolution_stats.single()["evolved_agents"]
                print(f"Agents That Have Evolved: {evolved_count}")
                
        except Exception as e:
            print(f"❌ Status check failed: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up enhanced system resources...")
        try:
            driver.close()
            data_driven_agent_manager.cleanup()
            gemma_text_to_cypher.cleanup()
            logger.info("Enhanced system cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Enhanced main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Self-Evolving AI Infrastructure")
    parser.add_argument("--mode", choices=["interactive", "demo", "test"], 
                       default="interactive", help="Run mode")
    parser.add_argument("--no-init", action="store_true", 
                       help="Skip system initialization")
    
    args = parser.parse_args()
    
    orchestrator = EnhancedOrchestrator()
    orchestrator.setup_signal_handlers()
    
    try:
        # Initialize enhanced system unless skipped
        if not args.no_init:
            success = await orchestrator.initialize_enhanced_system()
            if not success:
                logger.error("Enhanced system initialization failed, exiting...")
                sys.exit(1)
        
        # Run based on mode
        if args.mode == "interactive":
            await orchestrator.run_enhanced_interactive_mode()
        elif args.mode == "demo":
            await orchestrator._handle_evolution_demo()
        elif args.mode == "test":
            await orchestrator._handle_enhanced_status_command()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
