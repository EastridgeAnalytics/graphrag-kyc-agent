#!/usr/bin/env python3
"""
Quick Demo Script for the Self-Evolving AI Infrastructure

This script provides a simple way to run the evolution demonstration
and see the system in action.
"""

import asyncio
import sys
import time
from main_orchestrator import MainOrchestrator

async def main():
    """Run a streamlined evolution demonstration"""
    
    print("🧠 SELF-EVOLVING AI INFRASTRUCTURE DEMO")
    print("=" * 50)
    print("Initializing the living AI nervous system...")
    print("This system demonstrates agents that can:")
    print("  • Learn from their performance")
    print("  • Evolve their own instructions") 
    print("  • Replace poor performers")
    print("  • Self-organize in hierarchies")
    print()
    
    orchestrator = MainOrchestrator()
    
    try:
        # Initialize system
        print("🏗️ Initializing agent hierarchy...")
        success = await orchestrator.initialize_system()
        if not success:
            print("❌ System initialization failed!")
            return
        
        print("✅ System initialized successfully!")
        print()
        
        # Brief status check
        await orchestrator._handle_status_command()
        print()
        
        # Ask user if they want to run the full demo
        print("📊 CUSTOMER DEDUPLICATION EVOLUTION TEST")
        print("This demonstration will:")
        print("  1. Create 1000 customer records with duplicates")
        print("  2. Test 3 agents with different strategies")
        print("  3. Analyze performance and evolve poor performers")  
        print("  4. Validate improvements with fresh data")
        print("  5. Show 20%+ accuracy improvement")
        print()
        
        response = input("Run full evolution demonstration? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Demo cancelled. Use 'python main_orchestrator.py' for interactive mode.")
            return
        
        # Run the evolution demonstration
        print("\n🔬 Running Evolution Demonstration...")
        print("This may take 2-3 minutes...")
        
        start_time = time.time()
        results = await orchestrator.run_evolution_demonstration()
        duration = time.time() - start_time
        
        # Display results
        print(f"\n⚡ Demonstration completed in {duration:.1f} seconds")
        print("=" * 60)
        
        if results["success"]:
            summary = results["summary"]
            
            print("🎯 EVOLUTION RESULTS:")
            print(f"  Performance Improvement: {summary['performance_improvement']:+.1f}%")
            print(f"  Target Achievement: {'✅ MET' if summary['target_met'] else '❌ MISSED'} (>20% required)")
            print(f"  System Learning: {'✅ YES' if summary['system_learned'] else '❌ NO'}")
            print(f"  Agent Evolution: {'✅ YES' if summary['agents_evolved'] else '❌ NO'}")
            print()
            
            print("🏆 KEY ACHIEVEMENTS:")
            for achievement in summary["key_achievements"]:
                print(f"  ✅ {achievement}")
            
            print()
            print("🎉 SUCCESS! The AI system demonstrated self-evolution capabilities.")
            print("   Agents improved their own performance without human intervention.")
            
        else:
            print(f"❌ Demonstration failed: {results.get('error', 'Unknown error')}")
        
        print("\n📁 Results saved to demonstration_results_*.json")
        print("🔍 Use 'python main_orchestrator.py --mode interactive' for detailed exploration")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)
    finally:
        await orchestrator.cleanup()
        print("\n👋 Demo completed. System shut down gracefully.")

if __name__ == "__main__":
    asyncio.run(main())
