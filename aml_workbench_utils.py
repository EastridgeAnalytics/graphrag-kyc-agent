import os
import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv
from schemas import AlertModel
import pandas as pd
import uuid
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from streamlit_agraph import Node, Edge
from schemas import AlertModel, SARDraftModel


load_dotenv()

# Use st.cache_data to cache the driver connection
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password")
        )
    )

def get_alerts():
    driver = get_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run("MATCH (a:Alert) RETURN a")
        # Ensure that the node properties are accessed correctly
        alerts = [AlertModel(**record["a"]._properties) for record in result]
        return alerts

def get_customer_for_alert(alert_id):
    driver = get_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run("MATCH (c:Customer)-[:HAS_ALERT]->(a:Alert {id: $alert_id}) RETURN c.id as customer_id", alert_id=alert_id)
        record = result.single()
        return record["customer_id"] if record else None

def get_transactions_for_customer(customer_id):
    driver = get_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run("""
            MATCH (c:Customer {id: $customer_id})-[:OWNS]->(a:Account)
            MATCH (a)-[r:FROM|TO]->(t:Transaction)
            RETURN t.id as id, t.amount as amount, type(r) as type, t.timestamp as timestamp
            ORDER BY t.timestamp DESC
            LIMIT 20
        """, customer_id=customer_id)
        
        transactions = []
        for record in result:
            tx_type = record['type']
            amount = record['amount']
            transactions.append({
                'id': record['id'],
                'timestamp': record['timestamp'],
                'debit': amount if tx_type == 'FROM' else 0,
                'credit': amount if tx_type == 'TO' else 0,
            })
        
        if not transactions:
            return pd.DataFrame(columns=['timestamp', 'debit', 'credit']).set_index('timestamp')

        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df

def get_graph_for_alert(alert_id):
    driver = get_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run("""
            MATCH (alert:Alert {id: $alert_id})<-[:HAS_ALERT]-(customer:Customer)
            CALL apoc.path.subgraphAll(customer, {maxLevel: 2})
            YIELD nodes, relationships
            RETURN nodes, relationships
        """, alert_id=alert_id)
        
        record = result.single()
        if not record:
            return [], [], pd.DataFrame()

        nodes_raw = record["nodes"]
        rels_raw = record["relationships"]

        nodes = []
        edges = []
        node_details = []

        node_ids = set()

        for node in nodes_raw:
            node_id = str(node.id)
            if node_id not in node_ids:
                node_ids.add(node_id)
                labels = list(node.labels)
                label = labels[0] if labels else "Node"
                
                properties = dict(node)
                
                # Convert Neo4j DateTime objects to ISO strings for DataFrame compatibility
                for key, value in properties.items():
                    if hasattr(value, 'isoformat'):  # Neo4j DateTime objects
                        properties[key] = value.isoformat()
                
                # Use a display name for the node label, fallback to id
                display_label = properties.get('name', properties.get('id', node_id))
                nodes.append(Node(id=node_id, label=str(display_label), size=15, shape="dot"))
                
                detail = {"neo4j_id": node_id, "labels": labels}
                detail.update(properties)
                node_details.append(detail)


        for rel in rels_raw:
            edges.append(Edge(source=str(rel.start_node.id), 
                              target=str(rel.end_node.id), 
                              label=rel.type))

        df = pd.DataFrame(node_details)
        return nodes, edges, df


def generate_sar_narrative(analyst_commentary, graph_data_df):
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # Make sure OPENAI_API_KEY is in .env
    
    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert AML analyst. Your task is to write a Suspicious Activity Report (SAR) narrative.
        
        An analyst has provided their initial commentary, and you have been given a table of entities related to the alert.
        
        Analyst Commentary:
        {commentary}
        
        Entities from the Knowledge Graph:
        {graph_data}
        
        Based on this information, write a clear, concise, and comprehensive SAR narrative.
        The narrative should:
        1. Start with a summary of the suspicious activity.
        2. Detail the entities involved, referencing their IDs and properties from the table.
        3. Explain the connections between the entities and why they are suspicious.
        4. Conclude with a recommendation for filing a SAR.
        
        SAR Narrative:
        """
    )
    
    chain = prompt_template | llm
    
    graph_data_str = graph_data_df.to_string()
    
    response = chain.invoke({
        "commentary": analyst_commentary,
        "graph_data": graph_data_str
    })
    
    return response.content

def store_sar_draft(alert_id, analyst_commentary, narrative):
    driver = get_driver()
    sar_id = f"SAR_{uuid.uuid4().hex[:8].upper()}"
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        session.run("""
            MATCH (alert:Alert {id: $alert_id})
            CREATE (sar:SAR_Draft {
                id: $sar_id,
                narrative: $narrative,
                analyst_commentary: $analyst_commentary,
                created_at: datetime(),
                status: 'draft'
            })
            MERGE (sar)-[:BASED_ON]->(alert)
        """, alert_id=alert_id, sar_id=sar_id, analyst_commentary=analyst_commentary, narrative=narrative)
    return sar_id

def get_sar_drafts():
    driver = get_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run("MATCH (sar:SAR_Draft) RETURN sar ORDER BY sar.created_at DESC")
        
        drafts = []
        for record in result:
            properties = dict(record["sar"].items())
            # Convert Neo4j DateTime to ISO 8601 string for Pydantic
            if 'created_at' in properties and hasattr(properties['created_at'], 'isoformat'):
                properties['created_at'] = properties['created_at'].isoformat()
            
            drafts.append(SARDraftModel(**properties))
            
        return drafts

def get_graph_for_sar(sar_id):
    driver = get_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run("""
            MATCH (sar:SAR_Draft {id: $sar_id})-[:BASED_ON]->(alert:Alert)<-[:HAS_ALERT]-(customer:Customer)
            CALL apoc.path.subgraphAll(customer, {maxLevel: 2})
            YIELD nodes, relationships
            RETURN nodes, relationships
        """, sar_id=sar_id)
        
        record = result.single()
        if not record:
            return [], [], pd.DataFrame()

        nodes_raw = record["nodes"]
        rels_raw = record["relationships"]

        nodes = []
        edges = []
        node_details = []
        node_ids = set()

        for node in nodes_raw:
            node_id = str(node.id)
            if node_id not in node_ids:
                node_ids.add(node_id)
                labels = list(node.labels)
                properties = dict(node)
                
                # Convert Neo4j DateTime objects to ISO strings for DataFrame compatibility
                for key, value in properties.items():
                    if hasattr(value, 'isoformat'):  # Neo4j DateTime objects
                        properties[key] = value.isoformat()
                
                display_label = properties.get('name', properties.get('id', node_id))
                
                # Exclude the SAR_Draft node itself from the visualization
                if "SAR_Draft" not in labels:
                    nodes.append(Node(id=node_id, label=str(display_label), size=15, shape="dot"))
                
                detail = {"neo4j_id": node_id, "labels": labels}
                detail.update(properties)
                node_details.append(detail)

        for rel in rels_raw:
            # Ensure we don't add edges connected to the SAR_Draft node
            start_node_labels = list(rel.start_node.labels)
            end_node_labels = list(rel.end_node.labels)
            if "SAR_Draft" not in start_node_labels and "SAR_Draft" not in end_node_labels:
                 edges.append(Edge(source=str(rel.start_node.id), 
                                   target=str(rel.end_node.id), 
                                   label=rel.type))

        df = pd.DataFrame(node_details)
        return nodes, edges, df

def update_sar_status(sar_id, status):
    driver = get_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        session.run("""
            MATCH (sar:SAR_Draft {id: $sar_id})
            SET sar.status = $status
        """, sar_id=sar_id, status=status)
