#!/bin/bash

# Start script for the Self-Evolving AI Infrastructure

echo "🧠 SELF-EVOLVING AI INFRASTRUCTURE"
echo "=================================="
echo "Starting the living AI nervous system..."
echo

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Start Neo4j
echo "🏗️ Starting Neo4j database..."
docker-compose up -d neo4j

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to be ready..."
sleep 10

# Check if Neo4j is responding
while ! docker-compose exec neo4j cypher-shell -u neo4j -p password "RETURN 1;" &> /dev/null; do
    echo "   Still waiting for Neo4j..."
    sleep 5
done

echo "✅ Neo4j is ready!"
echo

# Ask what to run
echo "Choose how to start the system:"
echo "1) Quick Demo - Run evolution demonstration"
echo "2) Interactive Mode - Explore the system"
echo "3) Docker Mode - Run in container"
echo "4) Just Neo4j - Database only"
echo

read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "🚀 Starting quick demo..."
        python run_evolution_demo.py
        ;;
    2)
        echo "🎮 Starting interactive mode..."
        python main_orchestrator.py --mode interactive
        ;;
    3)
        echo "🐳 Starting in Docker..."
        docker-compose up --build agentic-ai
        ;;
    4)
        echo "💾 Neo4j is running at http://localhost:7474"
        echo "   Username: neo4j"
        echo "   Password: password"
        echo "   Use Ctrl+C to stop"
        docker-compose logs -f neo4j
        ;;
    *)
        echo "Invalid choice. Starting interactive mode..."
        python main_orchestrator.py --mode interactive
        ;;
esac
