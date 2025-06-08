# A GraphRAG Agent - Know-Your-Customer
A basic GraphRAG agent built with OpenAI Agent SDK, Neo4j and Neo4j MCP server


# Before You Start

Before running the KYC Agent, ensure you have the following prerequisites installed and configured:

1. **Python 3.13+**
   - Download and install Python 3.13
   - Verify installation with: `python --version`

2. **uv Package Manager**
   - [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
   - Verify installation with: `uv --version`

3. **Ollama**
   - Ideally, you need a device for 6GB GPU Memory (the model itself is about 4.1GB)
   - Install [Ollama] (https://ollama.com/download)
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - Pull the Text-to-Cypher model:
     ```bash
     ollama pull ed-neo4j/t2c-gemma3-4b-it-q8_0-35k
     ```
   
   - Verify the model is available and being served locally
   ```bash
     ollama ls
     ```
     You should see `pull ed-neo4j/t2c-gemma3-4b-it-q8_0-35k` listed (available locally)

4. **OpenAI Key**
You will be using the OpenAI Agent SDK with an OpenAI model, so you need an OPENAI_KEY

# **Neo4j**

The project includes a synthetic dataset of 10,000 Customers with rich details including:
- Customer accounts and transactions
- Device and IP address information
- Employer relationships
- Watchlist and PEP status

You have two options for loading this dataset:

## Option 1: Local Neo4j Docker Instance

1. Convert the backup file to a database folder and load it:
   ```bash
   docker run --interactive --tty --rm \
       --volume=./data:/data \
       --volume=./backup:/backup \
       --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
       neo4j/neo4j-admin:enterprise \
   neo4j-admin database load neo4j --from-path=/backup
   ```
2. Start a Neo4j docker container using the provided docker-compose.yml:
   ```bash
   docker compose up -d
   ```

## Option 2: Neo4j AuraDB Professional (Managed Instance)

1. Sign up for a [Free Trial of Neo4j AuraDB Professional](https://console.neo4j.io/)

2. Create a new database instance, Choose `AuraDB Professional (1GB Memory)`

3. Locate your instance `Backup and Restore` option:
![Backup and Restore in Aura Console](images/image1.png)
Upload the backup file `./backup/kyc-data.backup`

The backup will take a couple of minutes to restore. 
Wait for your instance status to be RUNNING.

The AuraDB Professional option provides a fully managed solution with automatic updates, backups, and scaling capabilities.


# **Prepare to Run the Agent**

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd kyc-agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   uv sync
   source .venv/bin/activate
   ```

3. Set up environment variables:
   
   Create a `.env` file with your Neo4j and OpenAI credentials:
   ```
   NEO4J_URI=<your-neo4j-uri> - leave blank if running docker. Otherwise take from  your Professional instance credentials
   NEO4J_USER=<your-username> - leave blank if running docker. Otherwise take from  your Professional instance credentials
   NEO4J_PASSWORD=<your-password> - leave blank if running docker. Otherwise take from  your Professional instance credentials
   NEO4J_DATABASE=neo4j
   OPENAI_API_KEY=sk-... - Your OpenAI key
   ```

# **Run the Agent**

Start the agent:
```bash
python kyc_agent.py
```

## Run the following sequence of questions

Test the agent tools with these example questions:

1. Get the database schema:
   ```
   What is the schema of the database?
   ```
   This test the `MCP Neo4j server get schema` to fetch the schema.

2. Find suspicious rings:
   ```
   Show me watchlisted customers involved in rings
   ```
   This tests the `find_customer_rings` tool.

3. Check shared addresses:
   ```
   For the first customer, find all his addresses and find if they are shared with other customers
   ```
   This tests both the `Text-to-Cypher Generation` tool and the `MCP Neo4j server exec read query` tool.

4. Recent transactions:
   ```
   Find more details about this customer (accounts and recent transactions)
   ```
   This tests the ` get_customer_and_accounts` tool. 


5. Store conversation summary:
   ```
   Write a 500-word summary of this conversation. Store it as a memory, make sure to link it to every account and transasction mentioned in the conversation
   ```
   This tests the `create_memory` tool.


