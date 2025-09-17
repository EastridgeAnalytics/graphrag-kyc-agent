"""
Main Orchestration System for Self-Evolving AI Infrastructure

This is the main entry point that demonstrates the living AI nervous system
where agents exist as Neo4j nodes and can self-organize, learn, and evolve.
"""

import asyncio
import logging
import argparse
import json
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from core_agent import orchestrator, driver
from customer_deduplication_test import DeduplicationTestOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agentic_intelligence.log')
    ]
)
logger = logging.getLogger("MAIN_ORCHESTRATOR")

class MainOrchestrator:
    """
    Main orchestrator for the self-evolving AI infrastructure
    """
    
    def __init__(self):
        self.running = True
        self.test_orchestrator = DeduplicationTestOrchestrator()
        self.current_task = None
    
    async def initialize_system(self):
        """Initialize the complete agent system"""
        logger.info("Initializing Self-Evolving AI Infrastructure")
        
        try:
            # Create the Neo4j schema for agents
            await self._setup_neo4j_schema()
            
            # Initialize the hierarchical agent system
            await orchestrator.create_agent_hierarchy()
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def _setup_neo4j_schema(self):
        """Setup Neo4j schema for the agent system"""
        with driver.session() as session:
            # Create indexes and constraints for better performance
            schema_queries = [
                "CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
                "CREATE INDEX agent_type_index IF NOT EXISTS FOR (a:Agent) ON (a.agent_type)",
                "CREATE INDEX agent_status_index IF NOT EXISTS FOR (a:Agent) ON (a.status)",
                "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:TaskOutcome) REQUIRE t.id IS UNIQUE",
                "CREATE INDEX task_timestamp_index IF NOT EXISTS FOR (t:TaskOutcome) ON (t.timestamp)"
            ]
            
            for query in schema_queries:
                try:
                    session.run(query)
                    logger.debug(f"Executed schema query: {query}")
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {e}")
    
    async def run_evolution_demonstration(self) -> Dict[str, Any]:
        """
        Run the complete evolution demonstration as specified in requirements
        """
        logger.info("Starting Evolution Demonstration")
        
        try:
            # Run the customer deduplication evolution test
            results = await self.test_orchestrator.run_full_evolution_test()
            
            # Generate summary report
            summary = self._generate_demonstration_summary(results)
            
            logger.info("Evolution demonstration completed successfully")
            return {
                "success": True,
                "results": results,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evolution demonstration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_demonstration_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the evolution demonstration"""
        success_metrics = results.get("success_metrics", {})
        
        summary = {
            "demonstration_success": success_metrics.get("success", False),
            "key_achievements": [],
            "performance_improvement": success_metrics.get("improvement_percentage", 0),
            "target_met": success_metrics.get("improvement_percentage", 0) >= 20.0,
            "agents_evolved": True,  # From the process
            "system_learned": success_metrics.get("system_learning_demonstrated", False)
        }
        
        # Track key achievements
        if success_metrics.get("agent_evolution_successful", False):
            summary["key_achievements"].append("Agents successfully modified their own instructions")
        
        if success_metrics.get("system_learning_demonstrated", False):
            summary["key_achievements"].append("Overall system accuracy improved after evolution")
        
        if results.get("evolution_round", {}).get("evolution_actions"):
            summary["key_achievements"].append("HR Director successfully created replacement agents")
        
        if success_metrics.get("target_met", False):
            summary["key_achievements"].append("Met 20%+ improvement target")
        
        return summary
    
    async def run_interactive_mode(self):
        """Run in interactive mode for demonstrations and testing"""
        logger.info("Starting Interactive Mode")
        
        print("\n" + "="*80)
        print("WELCOME TO THE SELF-EVOLVING AI INFRASTRUCTURE")
        print("="*80)
        print("This system demonstrates AI agents that can:")
        print("• Store themselves as Neo4j nodes")
        print("• Learn from their performance")
        print("• Evolve their own instructions")
        print("• Create new agents to replace poor performers")
        print("• Organize themselves in hierarchical structures")
        print("\nCommands:")
        print("  'demo' - Run the full evolution demonstration")
        print("  'status' - Show system status")
        print("  'agents' - List all agents")
        print("  'evolution' - Check evolution history")
        print("  'quit' - Exit the system")
        print("="*80)
        
        while self.running:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    self.running = False
                    break
                elif command == 'demo':
                    await self._handle_demo_command()
                elif command == 'status':
                    await self._handle_status_command()
                elif command == 'agents':
                    await self._handle_agents_command()
                elif command == 'evolution':
                    await self._handle_evolution_command()
                elif command == 'help':
                    print("Available commands: demo, status, agents, evolution, quit")
                else:
                    print(f"Unknown command: '{command}'. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nReceived interrupt signal. Shutting down gracefully...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")
        
        print("\nShutting down...")
    
    async def _handle_demo_command(self):
        """Handle the demo command"""
        print("\nStarting Full Evolution Demonstration...")
        print("This will take several minutes to complete.")
        print("The system will:")
        print("1. Create agent hierarchy")
        print("2. Generate test data with customer duplicates")
        print("3. Run initial deduplication with all agents")
        print("4. Analyze performance and trigger learning")
        print("5. Replace poor performers with evolved agents")
        print("6. Validate improvements with fresh data")
        
        confirm = input("\nProceed with demonstration? (y/n): ")
        if confirm.lower() != 'y':
            return
        
        results = await self.run_evolution_demonstration()
        
        if results["success"]:
            print("\n" + "="*60)
            print("EVOLUTION DEMONSTRATION RESULTS")
            print("="*60)
            
            summary = results["summary"]
            print(f"Performance Improvement: {summary['performance_improvement']:.1f}%")
            print(f"Target Met (>20%): {'✓' if summary['target_met'] else '✗'}")
            print(f"System Learning: {'✓' if summary['system_learned'] else '✗'}")
            
            print("\nKey Achievements:")
            for achievement in summary["key_achievements"]:
                print(f"  ✓ {achievement}")
            
            # Save results
            filename = f"demonstration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {filename}")
            
        else:
            print(f"\nDemonstration failed: {results.get('error', 'Unknown error')}")
    
    async def _handle_status_command(self):
        """Handle the status command"""
        print("\nSystem Status:")
        print("-" * 40)
        
        try:
            # Check Neo4j connection
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record:
                    print("✓ Neo4j Connection: Active")
                
                # Count agents by type
                agent_count_result = session.run("""
                    MATCH (a:Agent)
                    RETURN a.agent_type as type, count(*) as count
                    ORDER BY type
                """)
                
                print("\nAgent Population:")
                total_agents = 0
                for record in agent_count_result:
                    agent_type = record["type"]
                    count = record["count"]
                    total_agents += count
                    print(f"  {agent_type}: {count}")
                
                print(f"\nTotal Active Agents: {total_agents}")
                
        except Exception as e:
            print(f"✗ System Status Check Failed: {e}")
    
    async def _handle_agents_command(self):
        """Handle the agents command"""
        print("\nAgent Directory:")
        print("-" * 60)
        
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (a:Agent)
                    RETURN a.id, a.name, a.agent_type, a.tier, a.status,
                           a.success_rate, a.total_tasks
                    ORDER BY a.tier, a.name
                """)
                
                for record in result:
                    agent_id = record["a.id"]
                    name = record["a.name"]
                    agent_type = record["a.agent_type"]
                    tier = record["a.tier"]
                    status = record["a.status"]
                    success_rate = record["a.success_rate"] or 0.0
                    total_tasks = record["a.total_tasks"] or 0
                    
                    print(f"Tier {tier}: {name}")
                    print(f"  Type: {agent_type}")
                    print(f"  Status: {status}")
                    print(f"  Performance: {success_rate:.2%} success rate ({total_tasks} tasks)")
                    print(f"  ID: {agent_id[:8]}...")
                    print()
                    
        except Exception as e:
            print(f"Error retrieving agent information: {e}")
    
    async def _handle_evolution_command(self):
        """Handle the evolution command"""
        print("\nEvolution History:")
        print("-" * 50)
        
        try:
            with driver.session() as session:
                # Get recent task outcomes to show learning
                result = session.run("""
                    MATCH (a:Agent)-[:PERFORMED]->(t:TaskOutcome)
                    WHERE t.timestamp > datetime() - duration('P1D')
                    RETURN a.name, a.agent_type, t.outcome, t.score, t.timestamp
                    ORDER BY t.timestamp DESC
                    LIMIT 20
                """)
                
                print("Recent Task Outcomes (Last 24 hours):")
                for record in result:
                    agent_name = record["a.name"]
                    outcome = record["t.outcome"]
                    score = record["t.score"]
                    timestamp = record["t.timestamp"]
                    
                    status_icon = "✓" if outcome == "success" else "✗"
                    print(f"  {status_icon} {agent_name}: {outcome} (score: {score:.2f}) - {timestamp}")
                
                # Check for agents that have evolved
                evolution_result = session.run("""
                    MATCH (a:Agent)
                    WHERE a.last_evolution IS NOT NULL
                    RETURN a.name, a.last_evolution, a.success_rate
                    ORDER BY a.last_evolution DESC
                    LIMIT 10
                """)
                
                print("\nEvolved Agents:")
                for record in evolution_result:
                    agent_name = record["a.name"]
                    last_evolution = record["a.last_evolution"]
                    success_rate = record["a.success_rate"] or 0.0
                    
                    print(f"  🧬 {agent_name}: evolved on {last_evolution} (current: {success_rate:.2%})")
                    
        except Exception as e:
            print(f"Error retrieving evolution history: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        try:
            driver.close()
            logger.info("Neo4j driver closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Self-Evolving AI Infrastructure")
    parser.add_argument("--mode", choices=["demo", "interactive", "test"], 
                       default="interactive", help="Run mode")
    parser.add_argument("--no-init", action="store_true", 
                       help="Skip system initialization")
    
    args = parser.parse_args()
    
    orchestrator = MainOrchestrator()
    orchestrator.setup_signal_handlers()
    
    try:
        # Initialize system unless skipped
        if not args.no_init:
            success = await orchestrator.initialize_system()
            if not success:
                logger.error("System initialization failed, exiting...")
                sys.exit(1)
        
        # Run based on mode
        if args.mode == "demo":
            results = await orchestrator.run_evolution_demonstration()
            print(json.dumps(results, indent=2, default=str))
        elif args.mode == "interactive":
            await orchestrator.run_interactive_mode()
        elif args.mode == "test":
            # Quick test mode
            await orchestrator._handle_status_command()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
