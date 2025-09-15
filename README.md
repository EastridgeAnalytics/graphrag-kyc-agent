# A GraphRAG Agent - Know-Your-Customer
A basic GraphRAG agent built with OpenAI Agent SDK, Neo4j and Neo4j MCP server


# Before You Start

Before running the KYC Agent, ensure you have the following prerequisites installed and configured:

1. **Python 3.13+**
   - Download and install Python 3.13
   - Verify installation with:
   ```bash
   python --version
   ```
    

2. **uv Package Manager**
   - [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
   - Verify installation with: 
   ```bash
   uv --version
   ```

3. **Ollama**
   - Ideally, you need a device with 6GB GPU Memory (the model weights take about 4.1GB)
   - [Install Ollama](https://ollama.com/download)
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

You have two options to create a Free Neo4j database

## Option 1: Local Neo4j Docker Instance


Start a Neo4j docker container using the provided docker-compose.yml:
```bash
docker compose up -d
```

## Option 2: Neo4j AuraDB Free (Managed Instance)

1. Head over to [Neo4j AuraDB Console](https://console.neo4j.io/)

2. Create a new database instance, Choose `AuraDB Free`.
Make sure to download your credentials.

# **Prepare to Run the Agent**

1. Clone the repository:
   ```bash
   git clone https://github.com/neo4j-product-examples/graphrag-kyc-agent.git
   cd graphrag-kyc-agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

3. Set up environment variables:
   
   Create a `.env` file with your Neo4j and OpenAI credentials:
   ```
   OPENAI_API_KEY=sk-... - Your OpenAI key
   ```

   If you are using an AuraDB Free tier instance, find the downloaded credentials.
   Look at the NEO4J_URI and locate your instance id. 

   For example, If your instance id is `NEO4J_URI=neo4j+s://4469a679.databases.neo4j.io`
   
   Your instance id is `4469a679`.

   Add the following to your `.env` file:
   ```
   NEO4J_URI=neo4j+s://<YOUR_INSTANCE_ID>.databases.neo4j.io
   NEO4J_USERNAME=<YOUR_INSTANCE_ID>
   NEO4J_PASSWORD=<YOUR_PASSWORD>
   NEO4J_DATABASE=<YOUR_INSTANCE_ID>
   ```

# **Load the dataset**
```bash
python generate_kyc_dataset.py
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
   Show me 3 watchlisted customers involved in rings
   ```
   This tests the `find_customer_rings` tool.

3. Check shared addresses:
   ```
   For each of these customers, find their addresses and find out if they are shared with other customers
   ```
   This tests both the `Text-to-Cypher Generation` tool and the `MCP Neo4j server exec read query` tool.

4. Recent transactions:
   ```
   Find more details about the customer with the shared adddress. List accounts and recent transactions
   ```
   This tests the ` get_customer_and_accounts` tool. 


5. Store conversation summary:
   ```
   Write a 300-word summary of this investigation into this customer. Store it as a memory, make sure to link it to accounts and transasction mentioned in the conversation
   ```
   This tests the `create_memory` tool.

6. Velocity of transactions:
   ```
   I'm reviewing alert 716d, and i'm wondering whats going on. why was this alert created? is there any pii linking those customers prior to these new links connected via phone?
   ```
   This tests the `get_customer_and_accounts` tool.



# **Run the AML Analyst Workbench**

This project includes a Streamlit application that simulates a futuristic AML Analyst Workbench.

## **Setup**

1.  **Activate the virtual environment**:
    ```bash
    source .venv/bin/activate
    ```

2.  **Install all dependencies**:
    This command will install all the necessary packages for both the agent and the Streamlit application from the `pyproject.toml` file.
    ```bash
    uv sync
    ```

3.  **Ensure your `.env` file is configured** with your Neo4j and OpenAI credentials as described in the sections above.

4.  **Load the dataset**:
    Make sure your Neo4j database is populated with the sample data, including the alerts.
    ```bash
    python generate_kyc_dataset.py
    ```

## **Running the Application**

Once the setup is complete, you can launch the Streamlit application:

```bash
streamlit run graphrag-kyc-agent/aml_workbench.py
```

The application will open in your web browser, and you can start exploring the AML Analyst Workbench.



--------------------------------

We can absolutely connect to multiple data sources like SQLite, MongoDB, and PostgreSQL. The MCP Toolbox is specifically designed for this. It acts as a universal adapter, allowing a single agent to communicate with many different types of databases, each serving a distinct purpose.
Why Would a Bank Need So Many Databases?
It might seem complex, but in a large financial institution, using multiple specialized databases is not only common, it's a strategic necessity. Different databases are designed to solve different problems, and a bank's needs are incredibly diverse.
Here’s a realistic breakdown of why a bank would use the specific databases you mentioned, and others from the diagram:
PostgreSQL (or Cloud SQL / AlloyDB): The System of Record.
Use Case: This is the heart of core banking. It's a relational database (SQL) that is perfect for transactional integrity. It would store the definitive records for customer profiles, account balances, and transaction ledgers. Every debit and credit is recorded here with ACID compliance, ensuring data consistency and reliability. This is your source of truth.
MongoDB (NoSQL Document Store): The Customer 360 & Digital Experience Hub.
Use Case: Banks have a huge amount of unstructured or semi-structured data related to each customer. MongoDB is perfect for this. It would store things like customer support chat logs, uploaded documents (like passports or utility bills for KYC), user activity logs from the mobile app, and marketing communication preferences. It’s flexible and scales well for the massive volume of interaction data.
Neo4j (Graph Database): The Connections & Intelligence Engine.
Use Case: This is what we are using now. It’s built to understand relationships. Its primary job is fraud detection, AML, and identifying complex networks of influence. It answers questions like "Do these 100 new 'clean skin' customers share any non-obvious connections?" or "Is this new business secretly owned by a sanctioned individual through a series of shell companies?"
SQLite (Embedded Database): The Edge & Local Tooling.
Use Case: SQLite is small, fast, and file-based. You wouldn't run a core banking system on it, but it's used everywhere for specific tasks. For example, a data scientist might use it on their local machine to store and analyze a small, exported dataset. A mobile application might even use an encrypted SQLite database to cache non-sensitive data locally on a user's phone for performance. In our demo, the genai-toolbox uses it to simulate a local, legacy customer database.
BigQuery (or other Data Warehouses): The Analytics Powerhouse.
Use Case: You can't run complex, long-running analytical queries against your live transactional database (PostgreSQL) without slowing it down. Data from PostgreSQL, MongoDB, and other sources is regularly copied into a data warehouse like BigQuery. This is where business intelligence teams run massive queries to analyze market trends, customer segmentation for marketing campaigns, and long-term risk modeling.
In this ecosystem, an agent powered by the MCP Toolbox is incredibly powerful. When investigating an alert in the graph (Neo4j), it can instantly pivot and ask:

```
"Show me this customer's full chat history from the last 30 days" (querying MongoDB).
"Pull the full, official transaction ledger for this account from the last 48 hours" (querying PostgreSQL).
"Has any other customer in the bank's history ever used this combination of device ID and address?" (querying BigQuery).
```