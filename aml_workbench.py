import streamlit as st
import pandas as pd
import pydeck as pdk
from aml_workbench_utils import (
    get_alerts, get_transactions_for_customer, get_customer_for_alert, 
    get_graph_for_alert, generate_sar_narrative, store_sar_draft,
    get_sar_drafts, get_graph_for_sar, update_sar_status
)
from streamlit_agraph import agraph, Node, Edge, Config


st.set_page_config(layout="wide")

st.title("AML Analyst Workbench")

# Initialize session state
if 'selected_alert' not in st.session_state:
    st.session_state.selected_alert = None
if 'investigated_alert_id' not in st.session_state:
    st.session_state.investigated_alert_id = None
if 'sar_narrative' not in st.session_state:
    st.session_state.sar_narrative = None
if 'selected_sar_id' not in st.session_state:
    st.session_state.selected_sar_id = None
if 'selected_sar_draft' not in st.session_state:
    st.session_state.selected_sar_draft = None

def rerun():
    st.session_state.clear()
    st.experimental_rerun()

tab1, tab2, tab3 = st.tabs(["Alert Overview", "Research and Analysis", "SAR Review"])

with tab1:
    st.header("Alert Overview")
    
    alerts = get_alerts()
    
    if alerts:
        st.subheader("New Alerts")
        
        num_columns = 4
        # Create a new row of columns for every 4 alerts
        for i in range(0, len(alerts), num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(alerts):
                    alert = alerts[i + j]
                    with cols[j]:
                        with st.container():
                            st.write(f"**Alert ID:** {alert.id}")
                            st.write(f"**Description:** {alert.description}")
                            st.write(f"**Timestamp:** {alert.timestamp}")
                            if st.button("Investigate", key=f"investigate_{alert.id}"):
                                st.session_state.selected_alert = alert

    if st.session_state.selected_alert:
        st.divider()
        selected_alert = st.session_state.selected_alert
        st.subheader(f"Details for Alert: {selected_alert.id}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Login Location")
            
            # Create a DataFrame for the map
            map_df = pd.DataFrame({
                'lat': [selected_alert.latitude],
                'lon': [selected_alert.longitude]
            })

            view_state = pdk.ViewState(
                latitude=selected_alert.latitude,
                longitude=selected_alert.longitude,
                zoom=12,
                pitch=50,
            )
            
            layer = pdk.Layer(
                'HexagonLayer',
                data=map_df,
                get_position='[lon, lat]',
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            )
            
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "Login Location"}
            ))

        with col2:
            st.subheader("Recent Transaction History")
            customer_id = get_customer_for_alert(selected_alert.id)
            if customer_id:
                transactions_df = get_transactions_for_customer(customer_id)
                if not transactions_df.empty:
                    st.bar_chart(transactions_df[['debit', 'credit']])
                else:
                    st.write("No transactions found for the associated customer.")
            else:
                st.write("Could not find customer associated with this alert.")


with tab2:
    st.header("Research and Analysis")
    
    alert_id_input = st.text_input("Enter Alert ID to investigate")
    
    if st.button("Run Analysis", key="run_analysis"):
        st.session_state.investigated_alert_id = alert_id_input
        st.session_state.sar_narrative = None # Clear previous narrative
    
    if st.session_state.investigated_alert_id:
        alert_id = st.session_state.investigated_alert_id
        st.subheader(f"Graph Visualization for Alert: {alert_id}")
        
        nodes, edges, df = get_graph_for_alert(alert_id)
        
        if nodes:
            config = Config(width=1200, 
                            height=800, 
                            directed=True, 
                            physics=True, 
                            hierarchical=False,
                            nodeHighlightBehavior=True,
                            highlightColor="#F7A7A6",
                            collapsible=True,
                            node={'labelProperty':'label'},
                            link={'labelProperty': 'label', 'renderLabel': True}
                            )
            
            agraph(nodes=nodes, edges=edges, config=config)
            
            st.subheader("Entities in Graph")
            st.dataframe(df)
            
            st.subheader("Analyst Commentary")
            commentary = st.text_area("Add your notes and observations here...", height=200, key=f"commentary_{alert_id}")
            
            if st.button("Generate SAR Narrative", key=f"generate_sar_{alert_id}"):
                with st.spinner("Generating SAR narrative... This may take a moment."):
                    narrative = generate_sar_narrative(commentary, df)
                    st.session_state.sar_narrative = narrative
            
            if st.session_state.sar_narrative:
                st.subheader("Generated SAR Draft")
                narrative = st.session_state.sar_narrative
                st.text_area("SAR Narrative", value=narrative, height=400, key=f"narrative_text_{alert_id}")
                
                if st.button("Save SAR Draft", key=f"save_sar_{alert_id}"):
                    with st.spinner("Saving SAR draft..."):
                        sar_id = store_sar_draft(alert_id, commentary, narrative)
                        st.success(f"Successfully saved SAR Draft with ID: {sar_id}")
                        # Clear state
                        st.session_state.investigated_alert_id = None
                        st.session_state.sar_narrative = None
                        rerun()
        else:
            st.warning(f"No graph data found for Alert ID: {alert_id}")


with tab3:
    st.header("SAR Review")
    
    st.subheader("Pending SAR Drafts")
    
    sar_drafts = get_sar_drafts()
    
    if sar_drafts:
        # Use model_dump() for Pydantic v2
        sar_df = pd.DataFrame([draft.model_dump() for draft in sar_drafts])
        st.dataframe(sar_df)

        sar_id_input = st.text_input("Enter SAR_Draft ID to review")
        
        if st.button("Review SAR", key="review_sar"):
            st.session_state.selected_sar_id = sar_id_input
            # Find the selected SAR draft from the list
            selected_draft = next((draft for draft in sar_drafts if draft.id == sar_id_input), None)
            st.session_state.selected_sar_draft = selected_draft

    else:
        st.write("No SAR drafts found.")

    if st.session_state.selected_sar_id and st.session_state.selected_sar_draft:
        sar_id = st.session_state.selected_sar_id
        sar_draft = st.session_state.selected_sar_draft
        
        st.subheader(f"Reviewing SAR Draft: {sar_id}")

        nodes, edges, df = get_graph_for_sar(sar_id)

        if nodes:
            config = Config(width=1200, height=800, directed=True, physics=True, hierarchical=False, 
                            nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True,
                            node={'labelProperty':'label'}, link={'labelProperty': 'label', 'renderLabel': True})
            
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.warning("Could not retrieve graph for this SAR draft.")
        
        st.subheader("Analyst Commentary")
        st.text_area("Original Commentary", value=sar_draft.analyst_commentary, height=150, disabled=True, key=f"commentary_review_{sar_id}")
        
        st.subheader("Generated SAR Narrative")
        st.text_area("Narrative", value=sar_draft.narrative, height=300, disabled=True, key=f"narrative_review_{sar_id}")

        st.subheader("Decision")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("Accept SAR", key=f"accept_{sar_id}"):
                update_sar_status(sar_id, "accepted")
                st.success(f"SAR Draft {sar_id} has been accepted.")
                st.session_state.selected_sar_id = None
                st.session_state.selected_sar_draft = None
                rerun()

        with col2:
            if st.button("Reject SAR", key=f"reject_{sar_id}", type="primary"):
                update_sar_status(sar_id, "rejected")
                st.warning(f"SAR Draft {sar_id} has been rejected.")
                st.session_state.selected_sar_id = None
                st.session_state.selected_sar_draft = None
                rerun()
