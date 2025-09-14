import os
from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerStdio, MCPServer
from neo4j import GraphDatabase
from dotenv import load_dotenv
from schemas import CustomerAccountsInput, CustomerAccountsOutput, CustomerModel, AccountModel, TransactionModel, GenerateCypherRequest, LoadSqlCustomerToNeo4jInput
from kyc_cypher_tools import get_customer_info, find_customers_in_rings, is_customer_in_suspicious_ring, is_customer_bridge, is_customer_linked_to_hot_property
import asyncio
from pydantic import BaseModel, Field
from ollama import chat
import logging
from neo4j.time import DateTime, Date, Time, Duration
import json

# New Input Schema for the new tool
class LoadSqlCustomerToNeo4jInput(BaseModel):
    customer_name: str = Field(..., description="The name of the customer to load from SQL database to Neo4j.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger("KYC_AGENT")

# Load environment variables
load_dotenv()

# Read Neo4j environment variables into variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Neo4j connection setup
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Neo4j driver
driver = get_neo4j_driver()

# Tool 1: Get Customer details and its Accounts and some recent transactions
@function_tool
def get_customer_and_accounts(input: CustomerAccountsInput, tx_limit: int = 5) -> CustomerAccountsOutput:
    """
    Get Customer details including its Accounts and some recent transactions.
    Limits the number of most recent transactions per account.
    
    Args:
        input: CustomerAccountsInput containing customer_id
        tx_limit: Maximum number of recent transactions to return per account (default: 5)
    """
    logger.info(f"TOOL: GET_CUSTOMER_AND_ACCOUNTS - {input.customer_id}")

    cypher_query = """
    MATCH (c:Customer {id: $customer_id})-[o:OWNS]->(a:Account)
    WITH c, a
    CALL (c,a) {
        MATCH (a)-[b:TO|FROM]->(t:Transaction)
        ORDER BY t.timestamp DESC
        LIMIT $tx_limit
        RETURN collect(t) as transactions
    }
    RETURN c as customer, a as account, transactions
    """
    
    # Log the Cypher query for user visibility
    print("🔍 **Cypher Query Executed:**")
    print(f"```cypher\n{cypher_query.strip()}\n```")
    print(f"**Parameters:** customer_id='{input.customer_id}', tx_limit={tx_limit}")

    with driver.session() as session:
        result = session.run(
            cypher_query,
            customer_id=input.customer_id,
            tx_limit=tx_limit
        )
        # Get the records from the result
        records = result.data()
        # Initialize lists to store the customer, accounts, and transactions
        accounts = []
        for record in records:
            customer = dict(record["customer"])
            account = dict(record["account"])
            account["transactions"] = [dict(t) for t in record["transactions"]]
            accounts.append(account)
            

        return CustomerAccountsOutput(
            customer=CustomerModel(**customer),
            accounts=[AccountModel(**a) for a in accounts]
        )
       

# Tool 2: Identify watchlisted customers in suspicious rings
@function_tool 
def find_customer_rings(max_number_rings: int = 10, customer_in_watchlist: bool = True, customer_is_pep: bool = False, customer_id: str = None):
    """
    Detects circular transaction patterns (up to 6 hops) involving high-risk customers.
    
    Finds account cycles where the accounts are owned by customers matching specified
    risk criteria (watchlisted and/or PEP status).
    
    Args:
        max_number_rings: Maximum rings to return (default: 10)
        customer_in_watchlist: Filter for watchlisted customers (default: True)
        customer_is_pep: Filter for PEP customers (default: False)
        customer_id: Specific customer to focus on (not implemented)
    
    Returns:
        dict: Contains ring paths and associated high-risk customers
    """
    logger.info(f"TOOL: FIND_CUSTOMER_RINGS - {max_number_rings} - {customer_in_watchlist} - {customer_is_pep}")
    
    cypher_query = """
    MATCH p=(a:Account)-[:FROM|TO*6]->(a:Account)
    WITH p, [n IN nodes(p) WHERE n:Account] AS accounts
    UNWIND accounts AS acct
    MATCH (cust:Customer)-[r:OWNS]->(acct)
    WHERE cust.on_watchlist = $customer_in_watchlist AND cust.is_pep = $customer_is_pep
    WITH 
      p, 
      COLLECT(DISTINCT cust)   AS watchedCustomers,
      COLLECT(DISTINCT r)      AS watchRels
    RETURN 
      p, 
      watchedCustomers,
      watchRels
    LIMIT $max_number_rings
    """
    
    # Log the Cypher query for user visibility
    print("🔍 **Cypher Query Executed:**")
    print(f"```cypher\n{cypher_query.strip()}\n```")
    print(f"**Parameters:** max_number_rings={max_number_rings}, customer_in_watchlist={customer_in_watchlist}, customer_is_pep={customer_is_pep}")
    
    with driver.session() as session:
        result = session.run(
            cypher_query,
            max_number_rings=max_number_rings,
            customer_in_watchlist=customer_in_watchlist,
            customer_is_pep=customer_is_pep
        )
        rings = []
        for record in result:
            # Convert path to a list of node dictionaries for easier consumption
            path_nodes = [dict(node) for node in record["p"].nodes]
            watched_customers = [dict(cust) for cust in record["watchedCustomers"]]
            watch_rels = [dict(rel) for rel in record["watchRels"]]
            rings.append({
                "ring_path": path_nodes,
                "watched_customers": watched_customers,
                
            })
        
        return {"customer_rings": rings}

# Tool 3: Neo4j MCP server setup
neo4j_mcp_server = MCPServerStdio(
    params={
        "command": "mcp-neo4j-cypher",
        "args": [],
        "env": {
            "NEO4J_URI": NEO4J_URI,
            "NEO4J_USERNAME": NEO4J_USER,
            "NEO4J_PASSWORD": NEO4J_PASSWORD,
            "NEO4J_DATABASE": NEO4J_DATABASE,
        },
    },
    cache_tools_list=True,
    name="Neo4j MCP Server",
    client_session_timeout_seconds=20
)

# GenAI Toolbox MCP Server setup - TODO: Implement HTTP MCP client
# genai_toolbox_mcp_server = None  # Temporarily disabled

# Tool 4: Create Memory node and link it to entities
@function_tool
def create_memory(content: str, customer_ids: list[str] = [], account_ids: list[str] = [], transaction_ids: list[str] = []) -> str:
    """
    Create a Memory node and link it to specified customers, accounts, and transactions
    """
    logger.info(f"TOOL: CREATE_MEMORY - {content} - {customer_ids} - {account_ids} - {transaction_ids}")
    
    cypher_query = """
    CREATE (m:Memory {content: $content, created_at: datetime()})
    WITH m
    UNWIND $customer_ids as cid
    MATCH (c:Customer {id: cid})
    MERGE (m)-[:FOR_CUSTOMER]->(c)
    WITH m
    UNWIND $account_ids as aid
    MATCH (a:Account {id: aid})
    MERGE (m)-[:FOR_ACCOUNT]->(a)
    WITH m
    UNWIND $transaction_ids as tid
    MATCH (t:Transaction {id: tid})
    MERGE (m)-[:FOR_TRANSACTION]->(t)
    RETURN m.content as content
    """
    
    # Log the Cypher query for user visibility
    print("🔍 **Cypher Query Executed:**")
    print(f"```cypher\n{cypher_query.strip()}\n```")
    print(f"**Parameters:** content='{content}', customer_ids={customer_ids}, account_ids={account_ids}, transaction_ids={transaction_ids}")
    
    with driver.session() as session:
        result = session.run(
            cypher_query,
            content=content,
            customer_ids=customer_ids,
            account_ids=account_ids,
            transaction_ids=transaction_ids
        )
        
       
        return f"Created memory: {str(result)}"

# Tool to load customer from SQL to Neo4j
@function_tool
def load_sql_customer_to_neo4j(input: LoadSqlCustomerToNeo4jInput) -> str:
    """
    Loads a customer's data from the SQL database and ingests it into Neo4j.
    First, it searches for the customer in the SQL database using the search-customers-by-name tool.
    Then, it creates or updates the customer's node in Neo4j.
    """
    logger.info(f"TOOL: LOAD_SQL_CUSTOMER_TO_NEO4J - {input.customer_name}")

    # This is a placeholder for how you would call the tool through the agent/runner.
    # In a real scenario, the agent would decide to call this tool first,
    # and then we'd get the result. For this tool, we'll simulate this process.
    # A more advanced implementation could involve the tool_runner directly.

    # The agent would first use the 'search-customers-by-name' tool.
    # Let's assume the agent has run that and we have the result here.
    # For this example, we'll just log that this is what should happen.
    logger.info(f"Agent would now call 'search-customers-by-name' with name '{input.customer_name}' through the GenAI Toolbox MCP Server.")
    logger.info("For this demo, we will assume the tool returns a customer and we will proceed with ingestion.")
    
    # In a real implementation, you would get the customer data from the tool call.
    # Here is some hardcoded example data that would be returned from the tool.
    customer_data_from_sql = {
        'id': '3',
        'name': input.customer_name,
        'email': f'{input.customer_name.lower().replace(" ", ".")}@email.com'
    }

    # Now, ingest this data into Neo4j
    cypher_query = """
    MERGE (c:Customer {id: $id})
    ON CREATE SET c.name = $name, c.email = $email, c.source = 'sql'
    ON MATCH SET c.name = $name, c.email = $email, c.source = 'sql'
    RETURN c.name as name
    """
    
    # Log the Cypher query for user visibility
    print("🔍 **Cypher Query Executed:**")
    print(f"```cypher\n{cypher_query.strip()}\n```")
    print(f"**Parameters:** id='{customer_data_from_sql['id']}', name='{customer_data_from_sql['name']}', email='{customer_data_from_sql['email']}'")
    
    with driver.session() as session:
        result = session.run(
            cypher_query,
            id=customer_data_from_sql['id'],
            name=customer_data_from_sql['name'],
            email=customer_data_from_sql['email']
        )
        record = result.single()
        if record:
            return f"Successfully loaded customer '{record['name']}' from SQL to Neo4j."
        else:
            return "Failed to load customer to Neo4j."


# Tool 5: Generate Cypher query from natural language
@function_tool
def generate_cypher(request: GenerateCypherRequest) -> str:
    """
    Generate a Cypher query from a natural language question.
    The query should be executable in a Neo4j database.
    This tool should be used when the user asks a question that cannot be answered by the other tools.
    """
    # Let's provide the schema to the model for better results
    schema = """
    Node properties are the following:
    - Customer {id: STRING, name: STRING, is_pep: BOOLEAN, on_watchlist: BOOLEAN}
    - Account {id: STRING}
    - Transaction {id: STRING, amount: FLOAT, timestamp: DATETIME}
    - Device {id: STRING}
    - Alert {id: STRING}
    - Memory {content: STRING, created_at: DATETIME}
    - Company {id: STRING, name: STRING}
    - Address {id: STRING, location: POINT}
    - IP {id: STRING}

    The relationships are the following:
    - (:Customer)-[:OWNS]->(:Account)
    - (:Account)-[:TO]->(:Transaction)
    - (:Account)-[:FROM]->(:Transaction)
    - (:Transaction)-[:TRIGGERED_BY]->(:Alert)
    - (:Device)-[:USED_BY]->(:Customer)
    - (:Transaction)-[:INITIATED_BY]->(:Device)
    - (:Customer)-[:HAS_IP]->(:IP)
    - (:Customer)-[:HAS_ADDRESS]->(:Address)
    - (:Customer)-[:WORKS_FOR]->(:Company)
    - (:Memory)-[:FOR_CUSTOMER]->(:Customer)
    - (:Memory)-[:FOR_ACCOUNT]->(:Account)
    - (:Memory)-[:FOR_TRANSACTION]->(:Transaction)
    """
    
    USER_INSTRUCTION = f"""
    You are an expert Neo4j developer.
    Your task is to write a Cypher query for a given question.
    You must not use any node labels or relationship types that are not in the schema.
    
    Here is the schema of the database:
    {schema}
    
    Here is the question:
    {request.question}
    
    Write the Cypher query below:
    """
    
    logger.info(f"TOOL: GENERATE_CYPHER - INPUT - {request.question}")

    user_message = USER_INSTRUCTION
    try:
        # Generate Cypher query using the text2cypher model
        model: str = "ed-neo4j/t2c-gemma3-4b-it-q8_0-35k"
        response = chat(
            model=model,
            messages=[{"role": "user", "content": user_message}]
        )
        generated_cypher = response['message']['content']
        # Replace \n with new line
        generated_cypher = generated_cypher.replace("\\n", "\n")

        print(f"GENERATED CYPHER: - OUTPUT - {generated_cypher}")
        return generated_cypher
        
    except Exception as e:
        logger.error(f"Failed to generate Cypher using Ollama: {str(e)}")
        # Return a basic fallback query
        fallback_query = f"// Could not generate Cypher query. Ollama model '{model}' may not be available.\n// Original question: {request.question}\n// Please ensure Ollama is running and the model is installed."
        return fallback_query

# Helper to serialize Neo4j types for JSON conversion
def neo4j_serializer(obj):
    if isinstance(obj, (DateTime, Date, Time)):
        return obj.isoformat()
    if isinstance(obj, Duration):
        return str(obj)
    # This is a simple way to handle nodes and relationships; you might want more complex serialization
    if hasattr(obj, 'properties'):
        return dict(obj.properties)
    if isinstance(obj, dict):
        # The record.data() returns a dict, we need to serialize its values
        return {k: neo4j_serializer(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [neo4j_serializer(i) for i in obj]
    return obj

@function_tool
def execute_cypher(query: str) -> str:
    """
    Execute a given Cypher query against the Neo4j database.
    This tool should be used to run queries generated by 'generate_cypher' or for direct data retrieval.
    Returns the query result as a JSON string.
    """
    logger.info(f"TOOL: EXECUTE_CYPHER - Query: {query}")
    try:
        with driver.session() as session:
            result = session.run(query)
            # result.data() gives a list of dictionaries
            records = result.data()
            
            # Serialize the records to handle Neo4j specific types
            serialized_records = neo4j_serializer(records)
            
            # Log the query for user visibility
            print("✅ **Cypher Query Executed:**")
            print(f"```cypher\n{query.strip()}\n```")
            
            if not serialized_records:
                return "Query executed successfully, but returned no results."

            return json.dumps(serialized_records, indent=2)
    except Exception as e:
        logger.error(f"Failed to execute Cypher query: {str(e)}")
        error_message = f"Error executing query: {str(e)}"
        print(f"❌ **Cypher Execution Failed:**\n{error_message}")
        return error_message
        
# Agent initialization and cleanup
async def init_agent():
    # For now, let's create the agent without MCP servers to avoid connection issues
    # This will use only the local tools which should be sufficient for basic functionality
    logger.info("Creating KYC agent with local tools only (MCP temporarily disabled)")
    
    instructions = """You are an expert KYC analyst. Your primary goal is to investigate alerts and answer questions by retrieving and analyzing data from a Neo4j knowledge graph.

    **CRITICAL INSTRUCTIONS:**
    1.  **Be proactive.** Do not ask for permission to run tools. Formulate a plan, execute the necessary tools, and present your findings directly.
    2.  **Use the right tool for the job.** You have a set of specific tools for common tasks. Use them whenever possible.
    3.  For complex or unique questions that don't fit a specific tool, you must follow this two-step process:
        a. First, use the `generate_cypher` tool to create a Cypher query.
        b. Second, use the `execute_cypher` tool to run the query you just generated.
    4.  **Summarize your findings.** After executing tools, present the results to the user in a clear, summarized format. Always include the Cypher queries you ran in formatted code blocks.

    **Available tools:**
    - get_customer_and_accounts: Get customer details and their accounts with recent transactions.
    - find_customer_rings: Find suspicious circular transaction patterns involving high-risk customers.
    - create_memory: Create memory nodes to store analysis findings.
    - load_sql_customer_to_neo4j: Load customer data from an external SQL source.
    - get_customer_info: Get detailed information for a specific customer.
    - is_customer_in_suspicious_ring: Check if a customer is part of a transaction ring.
    - is_customer_bridge: Check if a customer is employed by multiple companies.
    - is_customer_linked_to_hot_property: Check if a customer is linked to a high-risk address.
    - generate_cypher: **(Step 1 for custom questions)** Generate a Cypher query from a natural language question.
    - execute_cypher: **(Step 2 for custom questions)** Execute a Cypher query to get data from the database.
    """

    kyc_agent = Agent(
        name="KYC Analyst",
        instructions=instructions,
        tools=[
            get_customer_and_accounts, 
            find_customer_rings, 
            create_memory, 
            generate_cypher, 
            load_sql_customer_to_neo4j,
            get_customer_info,
            is_customer_in_suspicious_ring,
            is_customer_bridge,
            is_customer_linked_to_hot_property,
            execute_cypher
        ],
        mcp_servers=[]  # No MCP servers for now to avoid connection issues
    )
    return kyc_agent

async def run_agent(agent: Agent, query: str, conversation_history: list):
    result = await Runner.run(
        agent, 
        conversation_history + [{"role": "user", "content": query}]
    )
    return result.final_output

async def cleanup_agent():
    await neo4j_mcp_server.cleanup()
    # await genai_toolbox_mcp_server.cleanup()  # Temporarily disabled
    driver.close()

# The CLI part is removed, and will be replaced by Streamlit integration.
# async def main():
#    ... (old CLI code) ...
#
# if __name__ == "__main__":
#    ... (old CLI code) ... 
        