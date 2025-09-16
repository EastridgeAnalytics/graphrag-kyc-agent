#!/usr/bin/env python3
"""
Enhanced Demo Script - Showcasing Evolution in Action

This script demonstrates:
1. Gemma3-4B powered natural language to Cypher queries
2. Data-driven agents that can be modified through database operations
3. Real-time evolution watching
4. Agent modification on-the-fly
"""

import asyncio
import sys
import time
from enhanced_orchestrator import EnhancedOrchestrator

async def main():
    """Run enhanced evolution demonstration"""
    
    print("🧠 ENHANCED SELF-EVOLVING AI INFRASTRUCTURE")
    print("=" * 60)
    print("🚀 NEW CAPABILITIES:")
    print("  • Gemma3-4B for natural language → Cypher queries")
    print("  • Fully data-driven agents stored in Neo4j")
    print("  • Real-time agent modification through database")
    print("  • Advanced evolution with performance analysis")
    print()
    
    orchestrator = EnhancedOrchestrator()
    
    try:
        # Initialize enhanced system
        print("🏗️ Initializing enhanced system...")
        success = await orchestrator.initialize_enhanced_system()
        if not success:
            print("❌ Enhanced system initialization failed!")
            return
        
        print("✅ Enhanced system initialized successfully!")
        print()
        
        # Show what we can do
        print("🎯 DEMONSTRATION OPTIONS:")
        print("1. Natural Language Queries with Gemma3-4B")
        print("2. Real-time Agent Modification")
        print("3. Watch Evolution in Action")
        print("4. Interactive Mode")
        print()
        
        choice = input("Select demonstration (1-4) or press Enter for interactive: ").strip()
        
        if choice == "1":
            await demo_natural_language_queries(orchestrator)
        elif choice == "2":
            await demo_agent_modification(orchestrator) 
        elif choice == "3":
            await demo_evolution_watching(orchestrator)
        else:
            print("🎮 Starting Enhanced Interactive Mode...")
            await orchestrator.run_enhanced_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)
    finally:
        await orchestrator.cleanup()
        print("\n👋 Enhanced demo completed. System shut down gracefully.")

async def demo_natural_language_queries(orchestrator):
    """Demonstrate Gemma3-4B natural language queries"""
    
    print("\n🤖 GEMMA3-4B NATURAL LANGUAGE TO CYPHER DEMO")
    print("=" * 50)
    
    sample_questions = [
        "Show me all agents and their performance",
        "Which agents have evolved recently?", 
        "Find the best performing entity resolution agent",
        "Show me the organizational hierarchy",
        "What tasks were completed in the last hour?"
    ]
    
    print("📝 Sample questions you can ask:")
    for i, question in enumerate(sample_questions, 1):
        print(f"  {i}. {question}")
    print()
    
    while True:
        question = input("🔍 Ask a question (or 'quit' to exit): ").strip()
        if question.lower() in ['quit', 'exit', '']:
            break
        
        await orchestrator._handle_natural_language_query(question)
        print()

async def demo_agent_modification(orchestrator):
    """Demonstrate real-time agent modification"""
    
    print("\n🔧 REAL-TIME AGENT MODIFICATION DEMO")
    print("=" * 40)
    
    try:
        # Get an agent to modify
        from data_driven_agents import data_driven_agent_manager
        agents = await data_driven_agent_manager.get_agents_by_criteria()
        
        if not agents:
            print("No agents found. Creating some first...")
            await orchestrator._create_data_driven_hierarchy()
            agents = await data_driven_agent_manager.get_agents_by_criteria()
        
        print("🤖 Available agents to modify:")
        for i, agent in enumerate(agents[:5]):
            print(f"  {i+1}. {agent.name} ({agent.agent_type})")
        
        choice = input("\nSelect agent number to modify: ").strip()
        try:
            agent_index = int(choice) - 1
            selected_agent = agents[agent_index]
        except (ValueError, IndexError):
            print("Invalid selection")
            return
        
        print(f"\n🎯 Selected: {selected_agent.name}")
        print(f"Current parameters: {selected_agent.parameters}")
        
        # Demonstrate modification
        print("\n🔄 Let's modify this agent's match_threshold parameter...")
        
        # Show before
        print("📊 BEFORE modification:")
        await orchestrator._execute_cypher_query(f"""
            MATCH (a:Agent {{id: '{selected_agent.agent_id}'}})
            RETURN a.name, a.parameters, a.updated_at
        """)
        
        # Modify through database
        new_threshold = 0.75 if selected_agent.parameters.get('match_threshold', 0.8) != 0.75 else 0.85
        await data_driven_agent_manager.update_agent_in_database(
            selected_agent.agent_id,
            {"parameters": f'{{"match_threshold": {new_threshold}}}'}
        )
        
        print(f"\n✅ Modified match_threshold to {new_threshold}")
        
        # Show after
        print("\n📊 AFTER modification:")
        await orchestrator._execute_cypher_query(f"""
            MATCH (a:Agent {{id: '{selected_agent.agent_id}'}})
            RETURN a.name, a.parameters, a.updated_at
        """)
        
        print("\n🎉 Agent successfully modified through database operations!")
        
    except Exception as e:
        print(f"❌ Modification demo failed: {e}")

async def demo_evolution_watching(orchestrator):
    """Demonstrate watching evolution in action"""
    
    print("\n🧬 EVOLUTION WATCHING DEMO")
    print("=" * 30)
    
    try:
        from data_driven_agents import data_driven_agent_manager
        
        # Get entity resolution agents
        entity_agents = await data_driven_agent_manager.get_agents_by_criteria(
            agent_type="Entity_Resolution_Agent"
        )
        
        if len(entity_agents) < 2:
            print("Creating entity resolution agents for demo...")
            await orchestrator._create_data_driven_hierarchy()
            entity_agents = await data_driven_agent_manager.get_agents_by_criteria(
                agent_type="Entity_Resolution_Agent"
            )
        
        print(f"🤖 Watching {len(entity_agents)} Entity Resolution Agents evolve...")
        print("\n📊 BEFORE evolution:")
        
        # Show current state
        for agent in entity_agents[:3]:
            print(f"  • {agent.name}")
            print(f"    Instructions length: {len(agent.instructions)} chars")
            print(f"    Evolution count: {agent.get_property('evolution_count', 0)}")
        
        print("\n🔄 Triggering evolution for all agents...")
        
        # Trigger evolution
        for agent in entity_agents[:3]:
            print(f"  🧬 Evolving {agent.name}...")
            await agent.evolve()
            time.sleep(1)  # Dramatic pause
        
        print("\n📊 AFTER evolution:")
        
        # Show evolved state
        for agent in entity_agents[:3]:
            # Reload to get updated data
            evolved_agent = await data_driven_agent_manager.load_agent_from_database(agent.agent_id)
            print(f"  • {evolved_agent.name}")
            print(f"    Instructions length: {len(evolved_agent.instructions)} chars")
            print(f"    Evolution count: {evolved_agent.get_property('evolution_count', 0)}")
        
        print("\n🎉 Evolution complete! Agents have learned and adapted!")
        
        # Show in Neo4j
        print("\n🌐 Check Neo4j to see the evolution history:")
        print("    Query: MATCH (a:Agent) WHERE a.last_evolution IS NOT NULL RETURN a.name, a.last_evolution")
        
    except Exception as e:
        print(f"❌ Evolution demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
