import streamlit as st
import pandas as pd
import pydeck as pdk
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from aml_workbench_utils import (
    get_alerts, get_transactions_for_customer, get_customer_for_alert, 
    get_graph_for_alert, generate_sar_narrative, store_sar_draft,
    get_sar_drafts, get_graph_for_sar, update_sar_status
)
from streamlit_agraph import agraph, Node, Edge, Config
from kyc_agent import init_agent, run_agent, cleanup_agent
import asyncio


st.set_page_config(layout="wide")

st.title("AML Analyst Workbench")

# Initialize session state
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
async def initialize_agent():
    if st.session_state.kyc_agent is None:
        st.session_state.kyc_agent = await init_agent()

# Run the async function to initialize the agent
try:
    asyncio.run(initialize_agent())
except Exception as e:
    st.error(f"Failed to initialize KYC agent: {e}")


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
            # Combine debit and credit for the chart, or use an empty list
            if not transactions_df.empty:
                # Using max to represent activity, could be sum() or other logic
                transaction_history = (transactions_df['debit'] + transactions_df['credit']).tolist()
            else:
                transaction_history = []

            alert_dict = alert.model_dump()
            alert_dict['transaction_history'] = transaction_history
            alerts_data.append(alert_dict)
        
        alert_df = pd.DataFrame(alerts_data)
        # Add a selection column
        alert_df.insert(0, "select", False)

        # Use st.data_editor to make a clickable dataframe
        edited_df = st.data_editor(
            alert_df[['select', 'id', 'description', 'transaction_history']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "select": st.column_config.CheckboxColumn(required=True),
                "transaction_history": st.column_config.BarChartColumn(
                    "Recent Transactions",
                    y_min=0,
                ),
            },
            disabled=alert_df.columns.drop("select"),
            key="alert_editor"
        )

        # Find the selected row
        selected_row = edited_df[edited_df.select]
        if not selected_row.empty:
            selected_alert_id = selected_row.iloc[0]['id']
            # If selection changed, update state and rerun
            if st.session_state.selected_alert_id != selected_alert_id:
                st.session_state.selected_alert_id = selected_alert_id
                st.session_state.selected_sar_id = None
                st.session_state.selected_view = 'alert_analysis'
                # No rerun here, let the script continue and redraw the other elements

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
                st.session_state.selected_alert_id = None
                st.session_state.selected_view = 'sar_review'
                # No rerun here, let the script continue
        
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

    if st.session_state.selected_view == 'alert_analysis' and st.session_state.selected_alert_id:
        alert_id = st.session_state.selected_alert_id
        st.subheader(f"Graph Visualization for Alert: {alert_id}")
        
        nodes, edges, df = get_graph_for_alert(alert_id)
        if nodes:
            graph_tab, entities_tab = st.tabs(["Graph Visualization", "Entities in Graph"])

            with graph_tab:
                config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False,
                                nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True,
                                node={'labelProperty':'label'}, link={'labelProperty': 'label', 'renderLabel': True})
                agraph(nodes=nodes, edges=edges, config=config)

            with entities_tab:
                st.dataframe(df, use_container_width=True)

            # Create tabs for Analyst Commentary and KYC Copilot
            commentary_tab, copilot_tab = st.tabs(["Analyst Commentary", "🤖 KYC Copilot"])
            
            with commentary_tab:
                commentary = st.text_area("Add your notes...", height=150, key=f"commentary_{alert_id}")
                
                if st.button("Generate SAR Narrative", key=f"generate_sar_{alert_id}"):
                    with st.spinner("Generating SAR narrative..."):
                        # Use the dataframe 'df' already fetched for the graph
                        narrative = generate_sar_narrative(commentary, df)
                        st.session_state.sar_narrative[alert_id] = narrative
                
                if st.session_state.sar_narrative.get(alert_id):
                    st.subheader("Generated SAR Draft")
                    st.text_area("SAR Narrative", st.session_state.sar_narrative[alert_id], height=200, disabled=True)
                    
                    if st.button("Save SAR Draft", key=f"save_sar_{alert_id}"):
                        sar_id = store_sar_draft(alert_id, commentary, st.session_state.sar_narrative[alert_id])
                        st.success(f"Saved SAR Draft: {sar_id}")
                        # Clear the SAR narrative and reset state
                        del st.session_state.sar_narrative[alert_id]
                        st.session_state.selected_alert_id = None
                        st.session_state.selected_view = None
                        rerun()
            
            with copilot_tab:
                # Initialize conversation history for this alert if not exists
                if f"conversation_{alert_id}" not in st.session_state:
                    st.session_state[f"conversation_{alert_id}"] = []
                
                # Create a container for the chat history to make it scrollable
                chat_container = st.container(height=400)
                
                # Display chat messages within the container
                with chat_container:
                    for message in st.session_state[f"conversation_{alert_id}"]:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask the KYC agent about this alert...", key=f"chat_{alert_id}"):
                    # Add user message to chat history immediately for better UX
                    st.session_state[f"conversation_{alert_id}"].append({"role": "user", "content": prompt})

                    # Rerun to display the new user message inside the container
                    st.rerun()

                # This part needs to be outside the initial chat input check to run after the rerun
                # Check if there's a new user message to process
                if (f"conversation_{alert_id}" in st.session_state and 
                    st.session_state[f"conversation_{alert_id}"] and
                    st.session_state[f"conversation_{alert_id}"][-1]["role"] == "user"):
                    
                    # Get the latest user prompt
                    last_message = st.session_state[f"conversation_{alert_id}"][-1]
                    prompt = last_message["content"]

                    # Get AI response
                    with st.spinner("Analyzing..."):
                        # Ensure agent is initialized
                        if 'kyc_agent' not in st.session_state or st.session_state.kyc_agent is None:
                            try:
                                # Check if OPENAI_API_KEY is available
                                if not os.getenv("OPENAI_API_KEY"):
                                    st.error("⚠️ OpenAI API Key is missing! Please add OPENAI_API_KEY to your .env file to use the KYC Copilot.")
                                    st.stop()
                                
                                st.session_state.kyc_agent = asyncio.run(init_agent())
                            except Exception as e:
                                st.error(f"Failed to initialize agent: {str(e)}")
                                st.stop()
                        
                        try:
                            # Create a context-aware query for the alert
                            context_query = f"Analyzing Alert {alert_id}: {prompt}"
                            
                            # Pass the conversation history (excluding the last user message which is the current prompt)
                            history = st.session_state[f"conversation_{alert_id}"][:-1]
                            
                            response = asyncio.run(run_agent(
                                st.session_state.kyc_agent, 
                                context_query, 
                                history
                            ))
                            
                            if response and response.strip():
                                # Add assistant response to chat history
                                st.session_state[f"conversation_{alert_id}"].append({"role": "assistant", "content": response})
                            else:
                                st.session_state[f"conversation_{alert_id}"].append({"role": "assistant", "content": "The agent didn't provide a response. Please try again."})
                                
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            error_msg = f"Agent Error: {str(e)}"
                            st.error(error_msg)
                            
                            # Add error to history
                            st.session_state[f"conversation_{alert_id}"].append({"role": "assistant", "content": f"I encountered an error: {str(e)}."})
                    
                    # Rerun to display the assistant's response
                    st.rerun()

        else:
            st.warning(f"No graph data found for Alert ID: {alert_id}")

    elif st.session_state.selected_view == 'sar_review' and st.session_state.selected_sar_id:
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

    if st.session_state.selected_view == 'alert_analysis' and st.session_state.selected_alert_id:
        alert_id = st.session_state.selected_alert_id
        selected_alert = next((a for a in alerts if a.id == alert_id), None)

        if selected_alert:
            st.subheader(f"Login Location for Alert: {alert_id}")
            map_df = pd.DataFrame({'lat': [selected_alert.latitude], 'lon': [selected_alert.longitude]})
            st.map(map_df, zoom=11)

            with st.expander("Show Detailed Transactions"):
                st.subheader("Recent Transactions")
                customer_id = get_customer_for_alert(alert_id)
                if customer_id:
                    transactions_df = get_transactions_for_customer(customer_id)
                    if not transactions_df.empty:
                        st.bar_chart(transactions_df[['debit', 'credit']])
                    else:
                        st.write("No transactions for customer.")

    elif st.session_state.selected_view == 'sar_review' and st.session_state.selected_sar_id:
        sar_id = st.session_state.selected_sar_id
        selected_draft = next((d for d in sar_drafts if d.id == sar_id), None)

        if selected_draft:
            st.subheader(f"Reviewing SAR: {sar_id}")
            st.text_area("Original Commentary", value=selected_draft.analyst_commentary, height=150, disabled=True)
            st.text_area("Generated Narrative", value=selected_draft.narrative, height=300, disabled=True)

    else:
        st.info("Select an item from the queue to see details.")
