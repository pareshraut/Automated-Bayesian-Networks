import re
import streamlit as st
import bnlearn as bn
import graphviz

def extract_and_format_nodes(response):
    """
    Extract the JSON data from the response and return the formatted data.

    :param response: The backend response, expected to contain the specified JSON structure.
    :return: A dictionary with the formatted data.
    """
    # Assuming the response is a string and the desired JSON is encapsulated
    # within curly braces {}. This extracts the JSON part from the response.
    start_idx = response.find('{')
    end_idx = response.rfind('}') + 1
    json_data = response[start_idx:end_idx]

    # Convert the extracted string into a dictionary
    data = eval(json_data)

    formatted_data = {
        'Event Node': data.get('Event Node', []),
        'Opportunity Node': data.get('Opportunity Node', []),
        'Trigger Nodes': data.get('Trigger Nodes', []),
        'Mitigator Nodes': data.get('Mitigator Nodes', []),
        'Control Nodes': data.get('Control Nodes', []),
        'External Influence Nodes': data.get('External Influence Nodes', []),
        'Outcome Nodes': data.get('Outcome Nodes', [])
    }

    with st.sidebar.expander("Current Nodes"):
        selected_category = st.selectbox("Select Category", list(formatted_data.keys()))
        if selected_category:
            selected_items = formatted_data[selected_category]
            for item in selected_items:
                st.write(item)

def extract_and_format_edges(response):
    formatted_data = []
    start_idx = response.find('[')
    end_idx = response.rfind(']') + 1
    edge_data = response[start_idx:end_idx]
    edges = re.findall(r'\("([^"]*)", "([^"]*)"\)', edge_data)


    for edge in edges:
        formatted_data.append(f"{edge[0]} -> {edge[1]}")

    with st.sidebar.expander("Current Edges"):
        for item in formatted_data:
            st.write(item)
    
    with st.sidebar.expander("Current Network"):
        graph = graphviz.Digraph()
        for edge in edges:
            graph.edge(edge[0], edge[1])
        st.graphviz_chart(graph)




