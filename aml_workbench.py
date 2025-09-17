import streamlit as st
import pandas as pd
import pydeck as pdk
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from aml_workbench_utils import (
    get_alerts, get_transactions_for_customer, get_customer_for_alert, 
    get_graph_for_alert, get_graph_for_alerts, generate_sar_narrative, store_sar_draft,
    get_sar_drafts, get_graph_for_sar, update_sar_status
)
from streamlit_agraph import agraph, Node, Edge, Config
from kyc_agent import init_agent, run_agent, cleanup_agent
import asyncio
from connections import Neo4jConnection, GenAIToolboxConnection
import logging

# Configure logging for the UI
log = logging.getLogger("UI")
log.setLevel(logging.INFO)


# Set page config
st.set_page_config(page_title="AML Analyst Workbench", layout="wide")

# Add a sidebar for data source configuration
with st.sidebar:
    st.header("🔗 Data Source Connections")
    st.info("The KYC Copilot can connect to these external data sources on command.")

    # Establish the connection to the Neo4j Database
    conn_neo4j = st.connection("neo4j", type=Neo4jConnection)
    
    # Check Neo4j health using the connection object
    neo4j_healthy = conn_neo4j.get_health() == "connected"
    
    if neo4j_healthy:
        st.markdown("🟢 **Neo4j Graph Database**")
        st.caption("Status: Connected")
        
        with st.expander("Available Graph Queries"):
            tools = conn_neo4j.list_tools()
            if tools:
                for tool in tools:
                    st.caption(tool.get('name'))
            else:
                st.caption("No tools available.")
    else:
        st.markdown("🔴 **Neo4j Graph Database**")
        st.caption("Status: Not Connected")

    st.markdown("---")
    
    # Establish and check the connection to the GenAI Toolbox
    conn_toolbox = st.connection("genai_toolbox", type=GenAIToolboxConnection)
    genai_toolbox_healthy = conn_toolbox.get_health() == "connected"

    if genai_toolbox_healthy:
        st.markdown("🟢 **PostgreSQL Customer DB**")
        st.caption("Status: Connected (via GenAI Toolbox)")
        with st.expander("Available SQL Tools"):
            tools = conn_toolbox.list_tools()
            if tools:
                for tool in tools:
                    st.caption(tool.get('name'))
            else:
                st.caption("No tools available.")
    else:
        st.markdown("🔴 **PostgreSQL Customer DB**")
        st.caption("Status: Not Connected (via GenAI Toolbox)")

    st.markdown("---")
    st.image("https://dist.neo4j.com/wp-content/uploads/20210617085700/neo4j-logo-2021.svg", width=150)

st.title("AML Analyst Workbench")

# Initialize session state
if 'selected_alert_ids' not in st.session_state:
    st.session_state.selected_alert_ids = []
if 'selected_alert_id' not in st.session_state:
    st.session_state.selected_alert_id = None
if 'selected_sar_id' not in st.session_state:
    st.session_state.selected_sar_id = None
if 'selected_view' not in st.session_state:
    st.session_state.selected_view = None
# Robustly initialize sar_narrative as a dictionary to prevent errors
if 'sar_narrative' not in st.session_state or st.session_state.sar_narrative is None:
    st.session_state.sar_narrative = {}
# Force clear the agent to ensure fresh initialization with new config
st.session_state.kyc_agent = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


def rerun():
    # Preserve selection across reruns if needed, or clear state
    st.rerun()

# --- Agent Initialization ---
# We now initialize the agent on-demand inside the chat logic
# to ensure it gets the latest connection status.


# Main layout with three columns
left_col, main_col, right_col = st.columns([1, 2.5, 1.5])

# Left Panel: Investigation Queue & Agent Chat
with left_col:
    st.header("Investigation Tools")
    
    # Investigation Queue only (KYC Copilot moved to Analysis Canvas)
    # --- Alerts Section ---
    st.subheader("New Alerts")
    alerts = get_alerts()
    
    if alerts:
        # Prepare data with transaction history for BarChartColumn
        alerts_data = []
        for alert in alerts:
            customer_id = get_customer_for_alert(alert.id)
            transactions_df = get_transactions_for_customer(customer_id)
            transaction_history = []
            if not transactions_df.empty:
                transaction_history = (transactions_df['debit'] + transactions_df['credit']).tolist()

            alert_dict = alert.model_dump()
            alert_dict['transaction_history'] = transaction_history
            alerts_data.append(alert_dict)
        
        alert_df = pd.DataFrame(alerts_data)
        alert_df.insert(0, "select", False) # Re-add the select column
        
        # Create a snapshot of the dataframe before the editor for comparison
        original_df = alert_df.copy()

        edited_df = st.data_editor(
            alert_df, # Pass the full dataframe to the editor
            hide_index=True,
            use_container_width=True,
            column_config={
                "select": st.column_config.CheckboxColumn("Select", required=True),
                "id": st.column_config.TextColumn("ID"),
                "description": st.column_config.TextColumn("Description"),
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["new", "under_review", "closed", "escalated"],
                    required=True,
                ),
                "transaction_history": st.column_config.BarChartColumn(
                    "Recent Transactions",
                    y_min=0,
                ),
            },
            disabled=['id', 'description', 'transaction_history'],
            key="alert_editor_inline"
        )

        # Determine what changed after user interaction
        selection_changed = st.session_state.selected_alert_ids != edited_df[edited_df.select]['id'].tolist()
        status_changed_mask = (edited_df['status'] != original_df['status'])
        status_changed = status_changed_mask.any()

        # If status was changed, update the DB. This is the highest priority action.
        if status_changed:
            changed_rows = edited_df[status_changed_mask]
            with st.spinner(f"Updating {len(changed_rows)} alert(s)..."):
                for _, row in changed_rows.iterrows():
                    alert_id = row['id']
                    new_status = row['status']
                    try:
                        from neo4j import GraphDatabase
                        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                        NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
                        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
                        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                        
                        with driver.session() as session:
                            session.run("MATCH (a:Alert {id: $id}) SET a.status = $status", id=alert_id, status=new_status)
                        st.toast(f"Updated alert {alert_id} to {new_status}")
                    except Exception as e:
                        st.error(f"Failed to update alert {alert_id}: {e}")
            
            # Also update selection state before rerunning
            st.session_state.selected_alert_ids = edited_df[edited_df.select]['id'].tolist()
            st.rerun()
        
        # If only the selection changed, update the selection state and rerun
        elif selection_changed:
            st.session_state.selected_alert_ids = edited_df[edited_df.select]['id'].tolist()
            st.rerun()

    else:
        st.info("No new alerts.")

    # --- SARs Section ---
    st.subheader("Pending SAR Drafts")
    sar_drafts = get_sar_drafts()
    if sar_drafts:
        sars_df = pd.DataFrame([d.model_dump() for d in sar_drafts])
        sars_df.insert(0, "select", False)

        edited_sars_df = st.data_editor(
            sars_df[['select', 'id', 'status', 'created_at']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "select": st.column_config.CheckboxColumn(required=True),
            },
            disabled=sars_df.columns.drop("select"),
            key="sar_editor"
        )

        selected_sar_row = edited_sars_df[edited_sars_df.select]
        if not selected_sar_row.empty:
            selected_sar_id = selected_sar_row.iloc[0]['id']
            if st.session_state.selected_sar_id != selected_sar_id:
                st.session_state.selected_sar_id = selected_sar_id
                st.session_state.selected_alert_ids = [] # Clear alert selection
                st.session_state.selected_view = 'sar_review'
                st.rerun()
        
        # If selection is cleared, reset the view
        elif st.session_state.selected_sar_id is not None and selected_sar_row.empty:
             st.session_state.selected_sar_id = None
             st.session_state.selected_view = None
             st.rerun()
        
        # Conditionally display Accept/Reject buttons if a SAR is selected
        if st.session_state.selected_view == 'sar_review' and st.session_state.selected_sar_id:
            sar_id = st.session_state.selected_sar_id
            st.write(f"**Reviewing:** `{sar_id}`")
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("Accept SAR", key=f"accept_{sar_id}", use_container_width=True):
                    update_sar_status(sar_id, "accepted")
                    st.success(f"SAR {sar_id} accepted.")
                    st.session_state.selected_sar_id = None
                    st.session_state.selected_view = None
                    rerun()
            with b_col2:
                if st.button("Reject SAR", key=f"reject_{sar_id}", type="primary", use_container_width=True):
                    update_sar_status(sar_id, "rejected")
                    st.warning(f"SAR {sar_id} rejected.")
                    st.session_state.selected_sar_id = None
                    st.session_state.selected_view = None
                    rerun()

    else:
        st.info("No pending SARs.")

# Center Panel: The Canvas (Graph and Map)
with main_col:
    st.header("Analysis Canvas")

    if st.session_state.selected_alert_ids:
        selected_ids = st.session_state.selected_alert_ids
        
        if len(selected_ids) == 1:
            st.subheader(f"Graph Visualization for Alert: {selected_ids[0]}")
            nodes, edges, df = get_graph_for_alert(selected_ids[0])
        else:
            st.subheader(f"Combined Graph for {len(selected_ids)} Alerts")
            nodes, edges, df = get_graph_for_alerts(selected_ids)

        if nodes:
            graph_tab, entities_tab = st.tabs(["Graph Visualization", "Entities in Graph"])

            with graph_tab:
                config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False,
                                nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True,
                                node={'labelProperty':'label'}, link={'labelProperty': 'label', 'renderLabel': True})
                agraph(nodes=nodes, edges=edges, config=config)

            with entities_tab:
                st.dataframe(df, use_container_width=True)

            # Only show commentary and copilot tabs for a single alert selection
            if len(selected_ids) == 1:
                alert_id = selected_ids[0]
                commentary_tab, copilot_tab = st.tabs(["Analyst Commentary", "🤖 KYC Copilot"])
                
                with commentary_tab:
                    commentary = st.text_area("Add your notes...", height=150, key=f"commentary_{alert_id}")
                    
                    if st.button("Generate SAR Narrative", key=f"generate_sar_{alert_id}"):
                        with st.spinner("Generating SAR narrative..."):
                            narrative = generate_sar_narrative(commentary, df)
                            st.session_state.sar_narrative[alert_id] = narrative
                    
                    if st.session_state.sar_narrative.get(alert_id):
                        st.subheader("Generated SAR Draft")
                        st.text_area("SAR Narrative", st.session_state.sar_narrative[alert_id], height=200, disabled=True)
                        
                        if st.button("Save SAR Draft", key=f"save_sar_{alert_id}"):
                            sar_id = store_sar_draft(alert_id, commentary, st.session_state.sar_narrative[alert_id])
                            st.success(f"Saved SAR Draft: {sar_id}")
                            del st.session_state.sar_narrative[alert_id]
                            st.session_state.selected_alert_ids = []
                            rerun()
                
                with copilot_tab:
                    # Initialize conversation history for this specific alert
                    if f"conversation_{alert_id}" not in st.session_state:
                        st.session_state[f"conversation_{alert_id}"] = []

                    # Display conversation history
                    chat_container = st.container(height=300)
                    with chat_container:
                        for message in st.session_state[f"conversation_{alert_id}"]:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

                    # Handle chat input
                    if prompt := st.chat_input("Ask the KYC agent about this alert..."):
                        # Append user message to this alert's conversation
                        st.session_state[f"conversation_{alert_id}"].append({"role": "user", "content": prompt})
                        
                        # -- Agent Initialization on Demand --
                        try:
                            with st.spinner("Initializing KYC Agent..."):
                                kyc_agent = asyncio.run(
                                    init_agent(
                                        use_genai_toolbox=genai_toolbox_healthy,
                                        genai_toolbox_conn=conn_toolbox if genai_toolbox_healthy else None
                                    )
                                )
                                if kyc_agent is None:
                                    st.error("Fatal: Agent initialization returned None.")
                                    st.stop()
                        except Exception as e:
                            st.error(f"Failed to initialize agent: {e}")
                            st.stop()

                        # -- Run Agent --
                        with st.spinner("Thinking..."):
                            try:
                                response = asyncio.run(run_agent(kyc_agent, prompt, st.session_state[f"conversation_{alert_id}"]))
                                st.session_state[f"conversation_{alert_id}"].append({"role": "assistant", "content": response})
                                st.rerun() # Rerun to display the new messages
                            except Exception as e:
                                st.error(f"I encountered an error: {e}")
            else:
                st.info("Analyst tools (Commentary, Copilot) are available when a single alert is selected.")

        else:
            st.warning(f"No graph data found for the selected alert(s).")

    elif st.session_state.selected_sar_id:
        sar_id = st.session_state.selected_sar_id
        st.subheader(f"Graph for SAR: {sar_id}")
        nodes, edges, df = get_graph_for_sar(sar_id)
        if nodes:
            config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False,
                            nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True,
                            node={'labelProperty':'label'}, link={'labelProperty': 'label', 'renderLabel': True})
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.warning("Could not retrieve graph for this SAR draft.")
    else:
        st.info("Select an Alert or SAR from the queue to begin analysis.")


# Right Panel: Contextual Details and Actions
with right_col:
    st.header("Details & Actions")

    if st.session_state.selected_alert_ids:
        selected_ids = st.session_state.selected_alert_ids
        # Get the full alert objects for the selected IDs
        selected_alerts = [a for a in alerts if a.id in selected_ids]

        if selected_alerts:
            if len(selected_alerts) == 1:
                st.subheader(f"Login Location for Alert: {selected_alerts[0].id}")
            else:
                st.subheader(f"Login Locations for {len(selected_alerts)} Alerts")

            # Create a DataFrame with coordinates for all selected alerts
            map_df = pd.DataFrame(
                [(alert.latitude, alert.longitude) for alert in selected_alerts],
                columns=['lat', 'lon']
            )
            st.map(map_df, zoom=2) # Use a wider zoom for multiple points

            # The expander for transactions is only shown for a single alert selection
            if len(selected_alerts) == 1:
                with st.expander("Show Detailed Transactions"):
                    st.subheader("Recent Transactions")
                    customer_id = get_customer_for_alert(selected_alerts[0].id)
                    if customer_id:
                        tab_sql, = st.tabs(["SQL (Postgres)"])


                        # SQL (Postgres) transactions via GenAI Toolbox
                        with tab_sql:
                            if genai_toolbox_healthy:
                                sql_resp = conn_toolbox.call_tool(
                                    tool_name="get-transactions-for-customer",
                                    args={"customer_id": customer_id}
                                )
                                sql_rows = sql_resp.get("data", []) if isinstance(sql_resp, dict) else []
                                # Prefer agent-fetched latest SQL data if present (keeps the UI in sync with Copilot)
                                latest_sql = st.session_state.get("last_sql_result")
                                if isinstance(latest_sql, list) and latest_sql:
                                    sql_rows = latest_sql
                                if sql_rows:
                                    sql_df = pd.DataFrame(sql_rows)
                                    st.dataframe(sql_df, use_container_width=True)
                                else:
                                    st.write("No SQL transactions for customer.")
                            else:
                                st.info("Postgres connection is not available.")

    elif st.session_state.selected_sar_id:
        sar_id = st.session_state.selected_sar_id
        selected_draft = next((d for d in sar_drafts if d.id == sar_id), None)

        if selected_draft:
            st.subheader(f"Reviewing SAR: {sar_id}")
            st.text_area("Original Commentary", value=selected_draft.analyst_commentary, height=150, disabled=True)
            st.text_area("Generated Narrative", value=selected_draft.narrative, height=300, disabled=True)

    else:
        st.info("Select an item from the queue to see details.")
