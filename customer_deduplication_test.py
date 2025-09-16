"""
Customer Deduplication Test Scenario

This module implements the proof-of-concept test for the self-evolving agent network.
It creates sample customer data with intentional duplicates and runs the evolution process.
"""

import asyncio
import logging
import random
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
import uuid
from core_agent import CoreAgent, AgentOrchestrator, orchestrator
from hr_director import HRDirectorAgent
from entity_resolution_agents import create_entity_resolution_agent

logger = logging.getLogger("DEDUPLICATION_TEST")

class CustomerDataGenerator:
    """
    Generates synthetic customer data with controlled duplicates for testing
    """
    
    def __init__(self):
        self.base_customers = self._create_base_customers()
        
    def _create_base_customers(self) -> List[Dict[str, Any]]:
        """Create the base set of 200 real customers"""
        base_customers = []
        
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Jennifer", 
                      "William", "Susan", "James", "Karen", "Christopher", "Nancy", "Daniel"]
        
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                     "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez"]
        
        streets = ["Main St", "Oak Ave", "Pine Rd", "Cedar Ln", "Elm Dr", "Maple Way", 
                  "First St", "Second Ave", "Park Rd", "Hill St"]
        
        cities = ["Springfield", "Franklin", "Georgetown", "Clinton", "Madison", "Washington", 
                 "Arlington", "Richmond", "Jackson", "Monroe"]
        
        for i in range(200):
            customer = {
                "id": f"cust_{i+1:03d}",
                "name": f"{random.choice(first_names)} {random.choice(last_names)}",
                "address": f"{random.randint(100, 9999)} {random.choice(streets)}, {random.choice(cities)}",
                "phone": f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
                "email": f"customer{i+1}@email.com"
            }
            base_customers.append(customer)
        
        return base_customers
    
    def generate_test_dataset(self, num_variations: int = 5) -> Tuple[List[Dict], List[List[str]]]:
        """
        Generate test dataset with duplicates and ground truth
        
        Returns:
            - List of customer records (including duplicates)
            - Ground truth groups (list of lists of customer IDs that should match)
        """
        all_customers = []
        ground_truth_groups = []
        
        for base_customer in self.base_customers:
            # Create variations of each base customer
            customer_group = [base_customer["id"]]
            variations = self._create_variations(base_customer, num_variations - 1)
            
            all_customers.append(base_customer)
            for variation in variations:
                all_customers.append(variation)
                customer_group.append(variation["id"])
            
            # Add ground truth group (all variations should be matched together)
            ground_truth_groups.append(customer_group)
        
        # Shuffle the dataset to make matching more challenging
        random.shuffle(all_customers)
        
        return all_customers, ground_truth_groups
    
    def _create_variations(self, base_customer: Dict, num_variations: int) -> List[Dict]:
        """Create variations of a base customer with different types of changes"""
        variations = []
        
        for i in range(num_variations):
            variation = base_customer.copy()
            variation["id"] = f"{base_customer['id']}_var_{i+1}"
            
            # Apply different types of variations
            variation_type = i % 4
            
            if variation_type == 0:
                # Name typos
                variation["name"] = self._add_name_typos(base_customer["name"])
            elif variation_type == 1:
                # Address variations
                variation["address"] = self._vary_address(base_customer["address"])
            elif variation_type == 2:
                # Phone format changes
                variation["phone"] = self._vary_phone_format(base_customer["phone"])
            elif variation_type == 3:
                # Multiple small changes
                variation["name"] = self._add_minor_name_change(base_customer["name"])
                variation["address"] = self._vary_address(base_customer["address"])
            
            variations.append(variation)
        
        return variations
    
    def _add_name_typos(self, name: str) -> str:
        """Add realistic typos to names"""
        typos = [
            # Common letter substitutions
            ("John", "Jon"), ("Michael", "Micheal"), ("Christopher", "Cristopher"),
            ("Katherine", "Catherine"), ("Steven", "Stephen")
        ]
        
        for original, typo in typos:
            if original in name:
                return name.replace(original, typo)
        
        # If no predefined typo, add random character change
        if len(name) > 4:
            pos = random.randint(1, len(name) - 2)
            name_list = list(name)
            name_list[pos] = random.choice("abcdefghijklmnopqrstuvwxyz")
            return "".join(name_list)
        
        return name
    
    def _add_minor_name_change(self, name: str) -> str:
        """Add minor changes like missing middle initial"""
        parts = name.split()
        if len(parts) == 2 and random.choice([True, False]):
            # Add middle initial
            return f"{parts[0]} {random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}. {parts[1]}"
        elif len(parts) == 3 and random.choice([True, False]):
            # Remove middle initial/name
            return f"{parts[0]} {parts[2]}"
        return name
    
    def _vary_address(self, address: str) -> str:
        """Create address variations"""
        variations = [
            ("Street", "St"), ("St", "Street"),
            ("Avenue", "Ave"), ("Ave", "Avenue"),
            ("Road", "Rd"), ("Rd", "Road"),
            ("Drive", "Dr"), ("Dr", "Drive"),
            ("Lane", "Ln"), ("Ln", "Lane")
        ]
        
        for original, replacement in variations:
            if original in address:
                return address.replace(original, replacement)
        
        return address
    
    def _vary_phone_format(self, phone: str) -> str:
        """Change phone number format"""
        # Remove formatting and reformat differently
        digits = ''.join(filter(str.isdigit, phone))
        if len(digits) == 10:
            formats = [
                f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
                f"{digits[:3]}.{digits[3:6]}.{digits[6:]}",
                f"{digits[:3]} {digits[3:6]} {digits[6:]}",
                digits
            ]
            return random.choice(formats)
        return phone

class DeduplicationTestOrchestrator:
    """
    Orchestrates the customer deduplication test scenario
    """
    
    def __init__(self):
        self.data_generator = CustomerDataGenerator()
        self.test_results = []
        self.evolution_history = []
    
    async def run_full_evolution_test(self) -> Dict[str, Any]:
        """
        Run the complete evolution test as specified in requirements
        """
        logger.info("Starting Customer Deduplication Evolution Test")
        
        # Step 1: Initialize the agent hierarchy
        await self._initialize_agents()
        
        # Step 2: Generate test data
        customer_records, ground_truth = self.data_generator.generate_test_dataset()
        logger.info(f"Generated {len(customer_records)} customer records with {len(ground_truth)} ground truth groups")
        
        # Step 3: Initial Round - All agents attempt deduplication
        initial_results = await self._run_initial_round(customer_records, ground_truth)
        
        # Step 4: Learning Round - Agents analyze performance
        learning_results = await self._run_learning_round()
        
        # Step 5: Evolution Trigger - Replace poor performers
        evolution_results = await self._run_evolution_round()
        
        # Step 6: Validation - Run test again with fresh dataset
        validation_results = await self._run_validation_round()
        
        # Step 7: Compile final results
        final_results = {
            "test_summary": {
                "total_records": len(customer_records),
                "ground_truth_groups": len(ground_truth),
                "test_timestamp": datetime.now().isoformat()
            },
            "initial_round": initial_results,
            "learning_round": learning_results,
            "evolution_round": evolution_results,
            "validation_round": validation_results,
            "success_metrics": self._calculate_success_metrics(initial_results, validation_results)
        }
        
        logger.info("Evolution test completed")
        return final_results
    
    async def _initialize_agents(self):
        """Initialize the agent hierarchy"""
        logger.info("Initializing agent hierarchy")
        
        # Create the core hierarchy
        await orchestrator.create_agent_hierarchy()
        
        # Get HR Director
        hr_agents = await orchestrator.get_agents_by_type("HR_Director_Agent")
        if not hr_agents:
            raise Exception("HR Director not found in hierarchy")
        
        # Convert to HRDirectorAgent if needed
        hr_director_core = hr_agents[0]
        from hr_director import HRDirectorAgent
        hr_director = HRDirectorAgent(
            agent_id=hr_director_core.agent_id,
            tier=hr_director_core.tier,
            instructions=hr_director_core.instructions,
            capabilities=hr_director_core.capabilities,
            parent_id=hr_director_core.parent_id
        )
        
        # Spawn Entity Resolution Agents
        entity_agent_ids = await hr_director.spawn_entity_resolution_agents()
        logger.info(f"Spawned {len(entity_agent_ids)} Entity Resolution Agents")
        
        return entity_agent_ids
    
    async def _run_initial_round(self, customer_records: List[Dict], ground_truth: List[List[str]]) -> Dict[str, Any]:
        """Run initial deduplication round with all agents"""
        logger.info("Running initial deduplication round")
        
        # Get all entity resolution agents
        entity_agents = await orchestrator.get_agents_by_type("Entity_Resolution_Agent")
        
        results = {}
        for agent_core in entity_agents:
            # Convert to proper entity resolution agent
            from entity_resolution_agents import create_entity_resolution_agent
            strategy = "balanced"  # Default strategy
            if "Conservative" in agent_core.name:
                strategy = "conservative"
            elif "Aggressive" in agent_core.name:
                strategy = "aggressive"
            
            agent = create_entity_resolution_agent(
                strategy=strategy,
                agent_id=agent_core.agent_id,
                agent_type=agent_core.agent_type,
                tier=agent_core.tier,
                instructions=agent_core.instructions,
                capabilities=agent_core.capabilities,
                parent_id=agent_core.parent_id
            )
            logger.info(f"Running deduplication with {agent.name}")
            
            task = {
                "type": "deduplicate",
                "customer_records": customer_records,
                "ground_truth": ground_truth
            }
            
            task_result = await agent.execute_task(task)
            
            if task_result.outcome.value == "success":
                details = task_result.details
                performance = details.get("performance_metrics", {})
                
                results[agent.agent_id] = {
                    "agent_name": agent.name,
                    "strategy": getattr(agent, 'matching_strategy', 'unknown'),
                    "groups_found": details.get("total_groups", 0),
                    "performance": performance,
                    "score": task_result.score
                }
            else:
                results[agent.agent_id] = {
                    "agent_name": agent.name,
                    "error": task_result.details.get("error", "Unknown error"),
                    "score": 0.0
                }
        
        # Rank agents by performance
        ranked_results = sorted(results.items(), key=lambda x: x[1].get("score", 0), reverse=True)
        
        return {
            "agent_results": results,
            "ranking": [(agent_id, data["agent_name"], data.get("score", 0)) for agent_id, data in ranked_results]
        }
    
    async def _run_learning_round(self) -> Dict[str, Any]:
        """Run learning round where agents analyze their performance"""
        logger.info("Running learning round")
        
        entity_agents = await orchestrator.get_agents_by_type("Entity_Resolution_Agent")
        learning_results = {}
        
        for agent in entity_agents:
            logger.info(f"Agent {agent.name} analyzing performance patterns")
            
            # Trigger agent evolution process
            await agent.evolve()
            
            learning_results[agent.agent_id] = {
                "agent_name": agent.name,
                "performance_before": agent.performance_metrics.average_score,
                "evolution_completed": True,
                "new_instructions_length": len(agent.instructions)
            }
        
        return {"learning_results": learning_results}
    
    async def _run_evolution_round(self) -> Dict[str, Any]:
        """Run evolution round - HR Director replaces poor performers"""
        logger.info("Running evolution round")
        
        # Get HR Director
        hr_agents = await orchestrator.get_agents_by_type("HR_Director_Agent")
        hr_director = hr_agents[0]
        
        # Conduct evolution review
        evolution_task = {
            "type": "evolution_review",
            "agent_type": "Entity_Resolution_Agent"
        }
        
        review_result = await hr_director.execute_task(evolution_task)
        
        evolution_actions = []
        if review_result.details.get("evolution_needed", False):
            recommendations = review_result.details.get("recommendations", [])
            
            for recommendation in recommendations:
                if recommendation["action"] == "replace_poor_performers":
                    # Replace poor performers
                    for poor_agent_id in recommendation["agents"]:
                        replace_task = {
                            "type": "replace_poor_performer",
                            "old_agent_id": poor_agent_id,
                            "successful_agent_ids": []  # Will be filled with successful agent IDs
                        }
                        
                        # Find successful agents for strategy copying
                        entity_agents = await orchestrator.get_agents_by_type("Entity_Resolution_Agent")
                        successful_agents = [a for a in entity_agents if a.performance_metrics.success_rate > 0.7]
                        replace_task["successful_agent_ids"] = [a.agent_id for a in successful_agents[:2]]
                        
                        replacement_result = await hr_director.execute_task(replace_task)
                        evolution_actions.append({
                            "action": "agent_replacement",
                            "old_agent_id": poor_agent_id,
                            "new_agent_id": replacement_result.details.get("new_agent_id"),
                            "success": replacement_result.outcome.value == "success"
                        })
        
        return {
            "evolution_needed": review_result.details.get("evolution_needed", False),
            "review_summary": review_result.details,
            "evolution_actions": evolution_actions
        }
    
    async def _run_validation_round(self) -> Dict[str, Any]:
        """Run validation with fresh dataset to measure improvement"""
        logger.info("Running validation round with fresh dataset")
        
        # Generate fresh test data
        fresh_records, fresh_ground_truth = self.data_generator.generate_test_dataset()
        
        # Run deduplication again with evolved agents
        validation_results = await self._run_initial_round(fresh_records, fresh_ground_truth)
        
        return {
            "fresh_dataset_size": len(fresh_records),
            "validation_results": validation_results
        }
    
    def _calculate_success_metrics(self, initial_results: Dict, validation_results: Dict) -> Dict[str, Any]:
        """Calculate success metrics comparing initial vs validation performance"""
        
        # Get average scores from initial round
        initial_scores = [data.get("score", 0) for data in initial_results["agent_results"].values()]
        initial_avg = sum(initial_scores) / len(initial_scores) if initial_scores else 0
        
        # Get average scores from validation round  
        validation_scores = [data.get("score", 0) for data in validation_results["validation_results"]["agent_results"].values()]
        validation_avg = sum(validation_scores) / len(validation_scores) if validation_scores else 0
        
        # Calculate improvement
        improvement_percentage = ((validation_avg - initial_avg) / initial_avg * 100) if initial_avg > 0 else 0
        
        return {
            "initial_average_score": initial_avg,
            "validation_average_score": validation_avg,
            "improvement_percentage": improvement_percentage,
            "target_improvement": 20.0,
            "success": improvement_percentage >= 20.0,
            "agent_evolution_successful": improvement_percentage > 0,
            "system_learning_demonstrated": validation_avg > initial_avg
        }

async def main():
    """Run the customer deduplication evolution test"""
    test_orchestrator = DeduplicationTestOrchestrator()
    
    try:
        results = await test_orchestrator.run_full_evolution_test()
        
        # Print results summary
        print("\n" + "="*80)
        print("CUSTOMER DEDUPLICATION EVOLUTION TEST RESULTS")
        print("="*80)
        
        success_metrics = results["success_metrics"]
        print(f"Initial Average Score: {success_metrics['initial_average_score']:.3f}")
        print(f"Final Average Score: {success_metrics['validation_average_score']:.3f}")
        print(f"Improvement: {success_metrics['improvement_percentage']:.1f}%")
        print(f"Target Met: {'✓' if success_metrics['success'] else '✗'} (>20% improvement)")
        print(f"System Learning: {'✓' if success_metrics['system_learning_demonstrated'] else '✗'}")
        
        # Save detailed results
        with open("evolution_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to evolution_test_results.json")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        from core_agent import driver
        driver.close()

if __name__ == "__main__":
    asyncio.run(main())
