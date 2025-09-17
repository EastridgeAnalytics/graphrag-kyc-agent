from streamlit.connections import ExperimentalBaseConnection
from streamlit.errors import StreamlitAPIException
import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from typing import Optional, Dict, Any
import httpx
import json

class GenAIToolboxConnection(ExperimentalBaseConnection):
    """
    A Streamlit connection to a GenAI Toolbox server, allowing for listing and calling tools.
    """

    def _connect(self, **kwargs) -> httpx.Client:
        """
        Connects to the GenAI Toolbox server.
        The URL is taken from Streamlit secrets ("genai_toolbox_url") or passed directly.
        """
        if "url" in kwargs:
            self._url = kwargs.pop("url")
        else:
            # Fallback to secrets if URL is not provided directly
            self._url = self._secrets.get("genai_toolbox_url", "http://localhost:5000")

        if not self._url:
            raise StreamlitAPIException("Missing GenAI Toolbox URL.")

        return httpx.Client(base_url=self._url, timeout=10.0)

    def get_health(self) -> str:
        """
        Checks the health of the GenAI Toolbox server by pinging its root endpoint.
        """
        try:
            response = self._instance.get("/")
            response.raise_for_status()
            # A successful response (e.g., 200 OK) means it's connected
            return "connected"
        except (httpx.RequestError, httpx.HTTPStatusError):
            return "disconnected"

    def list_tools(self) -> list:
        """
        Retrieves the list of available tools from the GenAI Toolbox server.
        It does this by fetching the default toolset.
        """
        try:
            response = self._instance.get("/api/toolset")
            response.raise_for_status()
            
            # The response is a toolset manifest, and the tools are the keys of the "tools" object
            tool_manifest = response.json().get("tools", {})
            
            # We need to reformat this into the simple list of dicts the UI expects
            tools_list = []
            for tool_name, tool_data in tool_manifest.items():
                tools_list.append({
                    "name": tool_name,
                    "description": tool_data.get("description", "No description available.")
                })
            return tools_list

        except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
            return []

    def call_tool(self, tool_name: str, args: dict) -> dict:
        """
        Calls a specific tool on the GenAI Toolbox server with the given arguments.
        """
        try:
            # The invoke endpoint is /api/tool/{toolName}/invoke
            response = self._instance.post(f"/api/tool/{tool_name}/invoke", json=args)
            response.raise_for_status()
            
            # The actual result is nested inside a "result" key and is a JSON string
            raw_response = response.json()
            if "result" in raw_response:
                try:
                    # The result is a JSON string that needs to be parsed
                    result_data = json.loads(raw_response["result"])
                    return {"data": result_data}
                except json.JSONDecodeError:
                    # If it's not JSON, return the raw string
                    return {"data": raw_response["result"]}
            return raw_response
            
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            # Attempt to parse the error response from the server for more clarity
            try:
                error_body = e.response.json()
                error_message = error_body.get("error", str(e))
            except (ValueError, AttributeError):
                error_message = str(e)
            return {"error": f"Failed to call tool '{tool_name}': {error_message}"}
        except ValueError:
            return {"error": f"Failed to decode JSON response from tool '{tool_name}'."}

class Neo4jConnection(ExperimentalBaseConnection):
    """
    A Streamlit connection to Neo4j for KYC graph data analysis.
    """

    def _connect(self, **kwargs) -> GraphDatabase.driver:
        """
        Connects to the Neo4j database.
        """
        uri = kwargs.get("uri", self._secrets.get("neo4j_uri", "bolt://localhost:7687"))
        username = kwargs.get("username", self._secrets.get("neo4j_username", "neo4j"))
        password = kwargs.get("password", self._secrets.get("neo4j_password", "password"))

        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test the connection
            driver.verify_connectivity()
            return driver
        except Exception as e:
            raise StreamlitAPIException(f"Failed to connect to Neo4j: {e}")

    def get_health(self) -> str:
        """
        Checks the health of the Neo4j connection.
        """
        try:
            with self._instance.session() as session:
                session.run("RETURN 1")
                return "connected"
        except Exception:
            return "disconnected"

    def query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None, ttl: int = 3600) -> pd.DataFrame:
        """
        Execute a Cypher query and return results as a DataFrame.
        """
        @st.cache_data(ttl=ttl)
        def _execute_cypher_cached(cypher: str, parameters: Optional[Dict[str, Any]]):
            try:
                with self._instance.session() as session:
                    result = session.run(cypher, parameters or {})
                    records = []
                    keys = result.keys()
                    
                    for record in result:
                        # Convert Neo4j record to dict, handling Node and Relationship objects
                        record_dict = {}
                        for key in keys:
                            value = record[key]
                            if hasattr(value, '_properties'):  # Node or Relationship
                                # Flatten node/relationship properties with prefix
                                for prop_key, prop_value in value._properties.items():
                                    record_dict[f"{key}_{prop_key}"] = prop_value
                                # Add labels for nodes
                                if hasattr(value, 'labels'):
                                    record_dict[f"{key}_labels"] = list(value.labels)
                            else:
                                record_dict[key] = value
                        records.append(record_dict)
                    
                    return pd.DataFrame(records)
            except Exception as e:
                st.error(f"Cypher query failed: {e}")
                return pd.DataFrame({"error": [str(e)]})

        return _execute_cypher_cached(cypher, parameters)

    def get_all_alerts(self, ttl: int = 3600) -> pd.DataFrame:
        """
        Get all alert nodes from Neo4j.
        """
        cypher = """
        MATCH (a:Alert)
        RETURN a.id as alert_id, a.description as description, 
               a.status as status, a.created_date as created_date
        ORDER BY a.id
        """
        return self.query(cypher, ttl=ttl)

    def get_alert_with_relationships(self, alert_id: str, ttl: int = 3600) -> pd.DataFrame:
        """
        Get an alert and its related entities (customers, transactions, etc).
        """
        cypher = """
        MATCH (a:Alert {id: $alert_id})
        OPTIONAL MATCH (a)-[r]-(related)
        RETURN a as alert, type(r) as relationship_type, related, labels(related) as related_labels
        """
        return self.query(cypher, {"alert_id": alert_id}, ttl=ttl)

    def search_customers_by_name(self, name: str, ttl: int = 3600) -> pd.DataFrame:
        """
        Search for customer nodes by name in Neo4j.
        """
        cypher = """
        MATCH (c:Customer)
        WHERE c.name CONTAINS $name
        RETURN c.id as customer_id, c.name as name, c.email as email,
               c.phone as phone, c.address as address
        ORDER BY c.name
        """
        return self.query(cypher, {"name": name}, ttl=ttl)

    def get_customer_network(self, customer_id: str, ttl: int = 3600) -> pd.DataFrame:
        """
        Get a customer's network (related customers, transactions, alerts).
        """
        cypher = """
        MATCH (c:Customer {id: $customer_id})
        OPTIONAL MATCH (c)-[r1]-(t:Transaction)-[r2]-(other:Customer)
        OPTIONAL MATCH (c)-[r3]-(a:Alert)
        RETURN c as customer, 
               collect(DISTINCT {transaction: t, other_customer: other}) as transactions,
               collect(DISTINCT a) as alerts
        """
        return self.query(cypher, {"customer_id": customer_id}, ttl=ttl)

    def get_suspicious_patterns(self, ttl: int = 3600) -> pd.DataFrame:
        """
        Find suspicious transaction patterns in the graph.
        """
        cypher = """
        MATCH (c1:Customer)-[:SENT]->(t:Transaction)-[:RECEIVED]->(c2:Customer)
        WHERE t.amount > 10000
        WITH c1, c2, count(t) as transaction_count, sum(t.amount) as total_amount
        WHERE transaction_count > 3
        RETURN c1.name as sender, c2.name as receiver, 
               transaction_count, total_amount
        ORDER BY total_amount DESC
        """
        return self.query(cypher, ttl=ttl)

    # For compatibility with existing UI code
    def list_tools(self) -> list:
        """
        Return available graph analysis operations.
        """
        return [
            {"name": "search-customers-by-name", "description": "Search for customers by name in the graph"},
            {"name": "get-all-alerts", "description": "Get all alerts from the graph"},
            {"name": "get-alert-details", "description": "Get detailed alert information with relationships"},
            {"name": "get-customer-network", "description": "Analyze a customer's network connections"},
            {"name": "get-suspicious-patterns", "description": "Find suspicious transaction patterns"}
        ]

    def call_tool(self, tool_name: str, args: dict) -> dict:
        """
        Call a graph analysis operation by name.
        """
        try:
            if tool_name == "search-customers-by-name":
                df = self.search_customers_by_name(args.get("name", ""))
                return {"data": df.to_dict("records")}
            elif tool_name == "get-all-alerts":
                df = self.get_all_alerts()
                return {"data": df.to_dict("records")}
            elif tool_name == "get-alert-details":
                df = self.get_alert_with_relationships(args.get("alert_id", ""))
                return {"data": df.to_dict("records")}
            elif tool_name == "get-customer-network":
                df = self.get_customer_network(args.get("customer_id", ""))
                return {"data": df.to_dict("records")}
            elif tool_name == "get-suspicious-patterns":
                df = self.get_suspicious_patterns()
                return {"data": df.to_dict("records")}
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}