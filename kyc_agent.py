# --- DEBUG HELPERS (paste once per file while debugging) ---
import traceback, logging
dbg = logging.getLogger("AGENT_DBG")
dbg.setLevel(logging.INFO)

def _assert_mcp_list(agent):
    ms = getattr(agent, "mcp_servers", None)
    # If the SDK leaves this as None when absent, that's OK. We only forbid [None] etc.
    if ms is None:
        return
    if not isinstance(ms, (list, tuple)):
        raise RuntimeError(f"mcp_servers must be list|tuple, got {type(ms)}")
    bad = [i for i, s in enumerate(ms) if s is None]
    if bad:
        where = "".join(traceback.format_stack(limit=8))
        raise RuntimeError(f"mcp_servers contains None at indexes {bad}\nStack:\n{where}")

def _dump_agent_state(agent, label=""):
    tools = [getattr(t, "name", repr(t)) for t in getattr(agent, "tools", [])]
    mcp = getattr(agent, "mcp_servers", None)
    dbg.info("AGENT_STATE %s tools=%s", label, tools)
    dbg.info("AGENT_STATE %s mcp_servers=%s", label, mcp)
# --- END DEBUG HELPERS ---


import os
import asyncio
import json
import httpx
import sys
import functools
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.time import DateTime, Date, Time, Duration
from ollama import chat

from agents import Agent, Runner, function_tool
from schemas import (
    CustomerAccountsInput, CustomerAccountsOutput, CustomerModel, AccountModel,
    TransactionModel, GenerateCypherRequest, SearchSqlCustomerInput,
    IngestSqlCustomerInput, GetTransactionsForAlertInput, UpdateAlertStatusInput,
    CustomerRingsOutput, RingModel, SearchSqlCustomerByRiskInput
)
from kyc_cypher_tools import (
    get_customer_info, find_customers_in_rings, is_customer_in_suspicious_ring,
    is_customer_bridge, is_customer_linked_to_hot_property, find_shared_pii_for_alert
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger("KYC_AGENT")

# Load environment variables
load_dotenv()

# Read Neo4j environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Neo4j connection setup
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

driver = get_neo4j_driver()

# Tool 1: Get Customer details and its Accounts and some recent transactions
@function_tool
def get_customer_and_accounts(input: CustomerAccountsInput, tx_limit: int = 5) -> CustomerAccountsOutput:
    """
    Get Customer details including its Accounts and some recent transactions.
    Limits the number of most recent transactions per account.
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

    print("🔍 **Cypher Query Executed:**")
    print(f"```cypher\n{cypher_query.strip()}\n```")
    print(f"**Parameters:** customer_id='{input.customer_id}', tx_limit={tx_limit}")

    with driver.session() as session:
        result = session.run(
            cypher_query,
            customer_id=input.customer_id,
            tx_limit=tx_limit
        )
        records = result.data()
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
def find_customer_rings(max_number_rings: int = 10, customer_in_watchlist: bool = True, customer_is_pep: bool = False, customer_id: str = None) -> CustomerRingsOutput:
    """
    Detect circular transaction patterns (up to 6 hops) involving high-risk customers.
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
            path_nodes = [dict(node) for node in record["p"].nodes]
            watched_customers = [dict(cust) for cust in record["watchedCustomers"]]
            watch_rels = [dict(rel) for rel in record["watchRels"]]
            rings.append(RingModel(
                ring_path=path_nodes,
                watched_customers=watched_customers,
                watch_relationships=watch_rels
            ))

        return CustomerRingsOutput(customer_rings=rings)

# Tool 3: Create Memory node and link it to entities
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

# Tool 4: Update alert status
@function_tool
def update_alert_status(input: UpdateAlertStatusInput) -> str:
    """
    Updates the status of a specific alert in the knowledge graph.
    """
    logger.info(f"TOOL: UPDATE_ALERT_STATUS - Alert ID: {input.alert_id}, New Status: {input.status}")

    cypher_query = """
    MATCH (a:Alert {id: $alert_id})
    SET a.status = $status
    RETURN a.id as alert_id, a.status as new_status
    """

    print("🔍 **Cypher Query Executed:**")
    print(f"```cypher\n{cypher_query.strip()}\n```")
    print(f"**Parameters:** alert_id='{input.alert_id}', status='{input.status}'")

    with driver.session() as session:
        result = session.run(cypher_query, alert_id=input.alert_id, status=input.status)
        record = result.single()
        if record:
            return f"Successfully updated status for alert '{record['alert_id']}' to '{record['new_status']}'."
        else:
            return f"Failed to update alert '{input.alert_id}'. It might not exist in the database."

# Tool 5: Generate Cypher query from natural language
@function_tool
def generate_cypher(request: GenerateCypherRequest) -> str:
    """
    Generate a Cypher query from a natural language question.
    """
    schema = """
    Node properties are the following:
    - Customer {id: STRING, name: STRING, is_pep: BOOLEAN, on_watchlist: BOOLEAN}
    - Account {id: STRING, name: STRING}
    - Company {id: STRING, name: STRING, industry: STRING}
    - Address {id: STRING, name: STRING, city: STRING, country: STRING}
    - Device {id: STRING, name: STRING, os: STRING}
    - IP_Address {id: STRING, name: STRING}
    - Payment_Method {id: STRING, name: STRING, pm_type: STRING, card_number: STRING}
    - Transaction {id: STRING, name: STRING, amount: FLOAT, timestamp: STRING}
    - Alert {id: STRING, description: STRING, timestamp: STRING, latitude: FLOAT, longitude: FLOAT, status: STRING, related_entity_id: STRING}
    - SAR_Draft {id: STRING}
    - PhoneNumber {id: STRING, name: STRING}
    - Memory {content: STRING, created_at: DATETIME}

    The relationships are the following:
    - (:Customer)-[:OWNS]->(:Account)
    - (:Customer)-[:EMPLOYED_BY]->(:Company)
    - (:Customer)-[:LIVES_AT]->(:Address)
    - (:Customer)-[:USES_DEVICE]->(:Device)
    - (:Device)-[:ASSOCIATED_WITH]->(:IP_Address)
    - (:Customer)-[:HAS_METHOD]->(:Payment_Method)
    - (:Account)-[:FROM]->(:Transaction)-[:TO]->(:Account)
    - (:Customer)-[:HAS_PHONE]->(:PhoneNumber)
    - (:Customer)-[:HAS_ALERT]->(:Alert)
    - (:Memory)-[:FOR_CUSTOMER]->(:Customer)
    - (:Memory)-[:FOR_ACCOUNT]->(:Account)
    - (:Memory)-[:FOR_TRANSACTION]->(:Transaction)
    """

    USER_INSTRUCTION = f"""
    You are an expert Neo4j Cypher query writer.
    Use ONLY the provided schema. Pay attention to property keys and relationship directions.

    SCHEMA:
    ---
    {schema}
    ---

    USER QUESTION:
    {request.question}

    Write only the Cypher query:
    """

    logger.info(f"TOOL: GENERATE_CYPHER - INPUT - {request.question}")

    try:
        model: str = "ed-neo4j/t2c-gemma3-4b-it-q8_0-35k"
        response = chat(
            model=model,
            messages=[{"role": "user", "content": USER_INSTRUCTION}]
        )
        generated_cypher = response['message']['content'].replace("\\n", "\n")
        print(f"GENERATED CYPHER: - OUTPUT - {generated_cypher}")
        return generated_cypher
    except Exception as e:
        logger.error(f"Failed to generate Cypher using Ollama: {str(e)}")
        fallback_query = (
            f"// Could not generate Cypher query. Ollama model may not be available.\n"
            f"// Original question: {request.question}\n"
            f"// Please ensure Ollama is running and the model is installed."
        )
        return fallback_query

# Helper to serialize Neo4j types for JSON conversion
def neo4j_serializer(obj):
    if isinstance(obj, (DateTime, Date, Time)):
        return obj.isoformat()
    if isinstance(obj, Duration):
        return str(obj)
    if hasattr(obj, 'properties'):
        return dict(obj.properties)
    if isinstance(obj, dict):
        return {k: neo4j_serializer(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [neo4j_serializer(i) for i in obj]
    return obj

@function_tool
def execute_cypher(query: str) -> str:
    """
    Execute a given Cypher query against the Neo4j database.
    Returns the result as a JSON string or a status message.
    """
    logger.info(f"TOOL: EXECUTE_CYPHER - Query: {query}")
    try:
        with driver.session() as session:
            result = session.run(query)
            records = result.data()
            serialized_records = neo4j_serializer(records)

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
async def init_agent(use_genai_toolbox: bool = False, genai_toolbox_conn=None):
    logger.info(f"Initializing KYC agent with GenAI Toolbox enabled: {use_genai_toolbox}")

    base_instructions = """You are an expert KYC analyst. Your primary goal is to investigate alerts and answer questions by retrieving and analyzing data from multiple sources.

    CRITICAL INSTRUCTIONS:
    1. Be thorough and query all sources when relevant.
    2. Compare and present findings with source attribution.
    3. Prefer high-level tools first; fall back to generate_cypher + execute_cypher last.
    4. Be proactive and summarize results with queries shown."""

    tools = [
        get_customer_and_accounts,
        find_customer_rings,
        create_memory,
        generate_cypher,
        update_alert_status,
        get_customer_info,
        is_customer_in_suspicious_ring,
        is_customer_bridge,
        is_customer_linked_to_hot_property,
        find_shared_pii_for_alert,
        execute_cypher,
    ]

    if use_genai_toolbox and genai_toolbox_conn:
        @function_tool
        def search_sql_customer_by_name(input: SearchSqlCustomerInput) -> str:
            logger.info(f"TOOL: SEARCH_SQL_CUSTOMER_BY_NAME - Name: {input.name}")
            conn = genai_toolbox_conn
            if conn is None:
                return "Error: GenAI Toolbox connection is not available."
            try:
                result = conn.call_tool(tool_name="search-customers-by-name", args={"name": input.name})
                if result and "data" in result:
                    data = result["data"]
                    if data:
                        print("✅ **MCP Response:**")
                        print(f"```json\n{json.dumps(data, indent=2)}\n```")
                        try:
                            import streamlit as st  # type: ignore
                            st.session_state["last_sql_result"] = data
                            st.session_state["last_sql_tool"] = "search-customers-by-name"
                        except Exception:
                            pass
                        return f"Successfully found customer data in the SQL database: {json.dumps(data, indent=2)}"
                    return "No customers found with that name in the SQL database."
                elif result and "error" in result:
                    return f"Error from GenAI Toolbox: {result['error']}"
                return f"Received an unexpected response from the GenAI Toolbox: {result}"
            except Exception as e:
                msg = f"An unexpected error occurred while querying the SQL database: {e}"
                logger.error(msg)
                return msg

        @function_tool
        def search_sql_customer_by_risk_score(input: SearchSqlCustomerByRiskInput) -> str:
            logger.info(f"TOOL: SEARCH_SQL_CUSTOMER_BY_RISK_SCORE - Score: {input.risk_score}")
            conn = genai_toolbox_conn
            if conn is None:
                return "Error: GenAI Toolbox connection is not available."
            try:
                result = conn.call_tool(tool_name="search-customers-by-risk-score", args={"risk_score": input.risk_score})
                if result and "data" in result:
                    data = result["data"]
                    if data:
                        print("✅ **MCP Response:**")
                        print(f"```json\n{json.dumps(data, indent=2)}\n```")
                        try:
                            import streamlit as st  # type: ignore
                            st.session_state["last_sql_result"] = data
                            st.session_state["last_sql_tool"] = "search-customers-by-risk-score"
                        except Exception:
                            pass
                        return f"Successfully found customers with risk score >= {input.risk_score}: {json.dumps(data, indent=2)}"
                    return f"No customers found with risk score >= {input.risk_score} in the SQL database."
                elif result and "error" in result:
                    return f"Error from GenAI Toolbox: {result['error']}"
                return f"Received an unexpected response from the GenAI Toolbox: {result}"
            except Exception as e:
                msg = f"An unexpected error occurred while querying the SQL database: {e}"
                logger.error(msg)
                return msg

        tools.append(search_sql_customer_by_name)
        tools.append(search_sql_customer_by_risk_score)
        instructions = base_instructions + "\n- SQL search tools enabled via GenAI Toolbox."
        logger.info("GenAI Toolbox (PostgreSQL) tool has been enabled.")
    else:
        instructions = base_instructions

    kyc_agent = Agent(
        name="KYC Analyst",
        instructions=instructions,
        tools=tools
    )
    _assert_mcp_list(kyc_agent)
    _dump_agent_state(kyc_agent, label="post-construct")
    return kyc_agent

async def run_agent(agent: Agent, query: str, conversation_history: list):
    if agent is None:
        raise RuntimeError("run_agent() received agent=None")
    _dump_agent_state(agent, label="pre-run")
    result = await Runner.run(
        agent,
        conversation_history + [{"role": "user", "content": query}]
    )
    return result.final_output

async def cleanup_agent():
    driver.close()

# CLI removed; Streamlit will own execution.
