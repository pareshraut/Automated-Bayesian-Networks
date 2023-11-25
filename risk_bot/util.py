import re
import streamlit as st
from html import escape
import ast
from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import datetime
import os
import pandas as pd
from multiprocessing import Pool
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MmhcEstimator
from pgmpy.factors.discrete import TabularCPD
from fredapi import Fred
import yfinance as yf 

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from joblib import Parallel, delayed
from pyvis import network as net
from sklearn.cluster import KMeans
import numpy as np

# Set the FRED API key
os.environ["FRED_API_KEY"] = "b73c11e70992c0501f8748360e192763"

# Category to color mapping
category_colors = {
    "Trigger Node": "#0766E2",  # True Blue
    "Risk Node": "#E20707",  # True Red
    "Mitigator Node": "#FFD60A",  # True Yellow
    "Outcome Node": "#42a45e"  # True Green
}


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
    start_index = response.find('[')
    end_index = response.find(']')
    edges_str = response[start_index:end_index+1]
    edges = ast.literal_eval(edges_str)

    with st.sidebar.expander("Current Edges"):
        edge_df = pd.DataFrame(edges, columns=["From", "To"])
        st.table(edge_df.style.hide(axis="index"))


    
    

def cpd_to_string(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    val = str(cpd)
    TabularCPD._truncate_strtable = backup
    return val

def get_empty_cpd(edges):
    start_idx = edges.find('[')
    end_idx = edges.rfind(']') + 1
    edge_data = edges[start_idx:end_idx]
    edges = re.findall(r'\("([^"]*)", "([^"]*)"\)', edge_data)

    # Create a Bayesian Network with your edges
    model = BayesianNetwork(edges)

    # Get a list of unique nodes from edges
    nodes = set(node for edge in edges for node in edge)

    # Create an empty CPD for each node, assuming 2 states for each node
    for node in nodes:
        parent_nodes = [parent for parent, child in edges if child == node]
        parent_cards = [2] * len(parent_nodes)  # assuming 2 states for each parent node
        if parent_nodes:
            values = [[0] * (2 ** len(parent_nodes))] * 2  # 2 rows for 2 states of the node, columns based on parent configurations
            cpd = TabularCPD(variable=node, variable_card=2, values=values, evidence=parent_nodes, evidence_card=parent_cards)
        else:
            cpd = TabularCPD(variable=node, variable_card=2, values=[[0], [0]])  # No parent nodes, corrected shape
        model.add_cpds(cpd)

    # Generate an empty CPD table as a string
    empty_cpd_str = ""

    for cpd in model.get_cpds():
        empty_cpd_str += cpd_to_string(cpd)  # Use str instead of to_string
        empty_cpd_str += "\n"

    return empty_cpd_str


def get_api(nodes):
    print('Inside get_api')
    examples = [
    {
        "input": '["Inflation", "Stop Loss Order", "Apple Stock Price"]',
        "output": '{"Inflation": ("FRED", "FPCPITOTLZGUSA"), "Stop Loss Order": ("User"), "Apple Stock Price": ("YFinance", "AAPL")}'
    },
    {
        "input": '["GDP"]',
        "output": '{"GDP": ("FRED", "GDP")}'
    },
    {
        "input": '["Exchange Rates", "Investment in safe assets"]',
        "output": '{"Exchange Rates": ("FRED", "EXUSEU"), "Investment in safe assets": ("User")}'
    },
    {
        "input": '["Price to Earnings Ratio"]',
        "output": '{"Price to Earnings Ratio": ("YFinance", "AAPL") }'
    },
    {
        "input": '["Inflation", "Investment in safe assets", "Trading Volume"]',
        "output": '{"Inflation": ("FRED", "FPCPITOTLZGUSA"), "Investment in safe assets": ("User"), "Trading Volume": ("YFinance", "AAPL")}'
    },
    {
        "input": '["Global Economic Outlook", "Government Policies", "Unemployment Rate"]',
        "output": '{"Global Economic Outlook": ("AlphaVantage", "global_outlook"), "Government Policies": ("AlphaVantage", "government_policies"), "Unemployment Rate": ("FRED", "UNRATE")}'
    },
    {
        "input": '["Oil Prices", "Stock Market Performance"]',
        "output": '{"Oil Prices": ("YFinance", "OIL"), "Stock Market Performance": ("YFinance", "SPY")}'
    },
    {
        "input": '["Consumer Sentiment", "Stock Performance: Gain or Loss"]',
        "output": '{"Consumer Sentiment": ("FRED", "UMCSENT"), "Stock Performance: Gain or Loss": "User"}'
    }
]


    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )


    final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI tasked with selecting the most suitable API for each of the following nodes' requirements. You can also suggest node adjacent ticker that would capture the sentiment. "
                   "If you believe that none of these APIs will provide the required data, please reply with 'User'. "
                   "Here are the available APIs and their purposes: "
                   "1. FRED: Use FRED for macro and microeconomic indicators. "
                   "2. YFinance: Choose YFinance for stock and commodity prices. "
                   "Please provide a list of nodes, and I will match each node with the most appropriate API, resulting in a dictionary mapping nodes to APIs and thier tickers or keywords. "
                   "Please return the data as a dictionary, where the key is the node and the value is a tuple of the API and the ticker or keyword."),
        few_shot_prompt,
        ("human", "{input}")
    ]
)


    chain = final_prompt | ChatOpenAI(temperature=0,openai_api_key='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',model_name='gpt-4')
    return chain.invoke({"input": nodes}).content

def get_data(api, ticker):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if api == 'FRED':
        fred = Fred()
        s = fred.get_series(ticker, observation_start="2023-01-01", observation_end= current_date)
        return s
    elif api == 'YFinance':
        if ticker == 'VIX':
            ticker = '^VIX'
        data = yf.download(ticker, start="2023-01-01", end=current_date)
        return data
    else:
        return None

def flatten_dict_values(d):
    def flatten(item):
        if isinstance(item, dict):
            for value in item.values():
                yield from flatten(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                yield from flatten(value)
        else:
            yield item

    return list(flatten(d))

def process_node(item):
    node_name, value = item
    try:    
        if isinstance(value, tuple) and len(value) == 2:
            api, ticker = value
            df = get_data(api, ticker)
            if api == 'FRED':
                df = pd.DataFrame(df, columns=[node_name])
            else:
                df = df.rename(columns={'Close': node_name})
                df = df[[node_name]]
            return df
    except ValueError:
        print('ValueError: ', node_name, value)
        pass
    
    return pd.DataFrame(columns=[node_name])

def get_data_from_nodes(nodes):
    print('Inside get_data_from_nodes')
    api_ticker = ast.literal_eval(get_api(flatten_dict_values(nodes)))
    print('api_ticker: ', api_ticker)
    
    dfs = []

    with Parallel(n_jobs=-1) as parallel:
        dfs = parallel(delayed(process_node)(item) for item in api_ticker.items())
    
    df = pd.concat(dfs, axis=1)
    
    # Step 1: Drop columns with all NaN values.
    df = df.dropna(axis=1, how='all')
    
    # Step 2: Forward fill NaN values in each column.
    for column in df.columns:
        df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
    
    # Step 3: Drop rows with any NaN values.
    #df = df.dropna(axis=0)

    for column in df.columns:
        num_unique = df[column].nunique()
        try:
            if num_unique >= 3:
                # Use K-Means clustering for three clusters
                kmeans = KMeans(n_clusters=3, random_state=0).fit(df[[column]])
                centroids = kmeans.cluster_centers_.flatten()
                labels = kmeans.labels_
                # Map labels based on centroids
                label_mapping = {np.argmin(centroids): 'Low', np.argmax(centroids): 'High'}
                middle_label = set(range(3)) - set(label_mapping.keys())
                label_mapping[middle_label.pop()] = 'Medium'
                df[column] = [label_mapping[l] for l in labels]
            elif num_unique == 2:
                # Use K-Means clustering for two clusters
                kmeans = KMeans(n_clusters=2, random_state=0).fit(df[[column]])
                centroids = kmeans.cluster_centers_.flatten()
                labels = kmeans.labels_
                # Map labels based on centroids
                label_mapping = {np.argmin(centroids): 'Low', np.argmax(centroids): 'High'}
                df[column] = [label_mapping[l] for l in labels]
            else:
                df = df.drop(columns=[column])
        except ValueError:
            print('ValueError: ', column, num_unique)
            df = df.drop(columns=[column])

    return df

# def get_edges_and_cpds(nodes):
#     print('Inside get_edges_and_cpds')
#     start_idx = nodes.find('{')
#     end_idx = nodes.rfind('}') + 1
#     json_data = nodes[start_idx:end_idx]
#     nodes = ast.literal_eval(json_data)
#     df = get_data_from_nodes(nodes)
#     est = MmhcEstimator(df)
#     model = est.estimate()
#     edges = model.edges()
#     print(edges)
#     bn = BayesianNetwork(edges)
#     bn.fit(df, estimator=MaximumLikelihoodEstimator)
#     cpd_strings = ''
#     for node in df.columns:
#         cpd = bn.get_cpds(node)
#         cpd_strings += cpd_to_string(cpd)
#         cpd_strings += 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
#     return edges, cpd_strings

def get_edges(nodes):
    """Compute the edges given the nodes."""
    print('Inside get_edges')
    start_idx = nodes.find('{')
    end_idx = nodes.rfind('}') + 1
    json_data = nodes[start_idx:end_idx]
    nodes = ast.literal_eval(json_data)
    df = get_data_from_nodes(nodes)
    est = MmhcEstimator(df)
    model = est.estimate()
    edges = model.edges()
    return edges

def get_cpds(edges):
    """Compute the CPDs given the edges and nodes."""
    print('Inside get_cpds')
    print('edges: ', edges)
    edges = ast.literal_eval(edges)
    df = get_data_from_nodes(edges)
    valid_edges = [edge for edge in edges if edge[0] in df.columns and edge[1] in df.columns]
    # Extract all nodes from valid_edges
    valid_nodes = list(set([node for edge in valid_edges for node in edge]))

    bn = BayesianNetwork(valid_edges)
    bn.fit(df, estimator=MaximumLikelihoodEstimator)
    cpd_strings = ''
    for node in valid_nodes:
        cpd = bn.get_cpds(node)
        cpd_strings += cpd_to_string(cpd)
        cpd_strings += 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    print('cpd_strings: ', cpd_strings)
    return cpd_strings

# def capture_multiple_dictionaries(input_string):
#     # Initialize variables to count '{' and '}' characters
#     start_index = 0
#     count = 0
#     dictionaries = []
    
#     # Iterate through the characters in the input string
#     for i, char in enumerate(input_string):
#         if char == '{':
#             if count == 0:
#                 start_index = i
#             count += 1
#         elif char == '}':
#             count -= 1
#             if count == 0:
#                 dictionaries.append(input_string[start_index:i + 1])
    
#     return dictionaries

def process_response(response):
    try:
        # response = response.replace('```json\n', '')
        # response = response.replace('```', '')
        node_start_index = response.find('{')
        node_end_index = response.rfind('}') + 1
        edge_start_index = response.find('[')
        edge_end_index = response.find(']')

        if node_start_index != -1 and node_end_index != -1:
            segments = []
            last_end = 0
            nodes_str = response[node_start_index:node_end_index+1]
            nodes = ast.literal_eval(nodes_str)
                
            # Append the text before nodes as the first segment
            if node_start_index > 0:
                segments.append(response[0:node_start_index])
            last_end = node_start_index + len(nodes_str)
            
            # Convert the dictionary to an HTML table with styling
            table = """
            <table class="styled-table">
            <thead>
                <tr><th>Category</th><th>Nodes</th></tr>
            </thead>
            <tbody>
                """
            
            for key, values in nodes.items():
                values_str = ", ".join(escape(value) for value in values)  # escape values to prevent HTML injection

                # Define information for each category
                category_info = {
                                    "Risk Node": "A 'Risk Node' in the Bayesian network represents an uncertain event or condition that may have negative consequences.",
                                    "Trigger Node": "A 'Trigger Node' in the Bayesian network represents an event or condition that can lead to the occurrence of another event.",
                                    "Mitigator Node": "A 'Mitigator Node' in the Bayesian network represents a factor or action that can reduce the impact or likelihood of a risk.",
                                    "Outcome Node": "An 'Outcome Node' in the Bayesian network represents the result or consequence of a set of events and conditions."
                                }

                category_description = category_info.get(key, "No information available for this category.")

                table += f"""<tr>
                    <td>{escape(key)}</td>
                    <td class="tooltip">
                        {values_str}
                        <span class="tooltiptext">{category_description}</span>
                    </td>
                </tr>
                """

            # Append the table as the next segment
            segments.append(table)
            
            # Append text after the nodes string
            segments.append(response[last_end:])
            
            # Return the combined segment
            return segments


        elif edge_start_index != -1 and edge_end_index != -1:
            segments = []
            last_end = 0
            edges_str = response[edge_start_index:edge_end_index+1]
            edges = ast.literal_eval(edges_str)
                
            # Append the text before edges as the first segment
            if edge_start_index > 0:
                segments.append(response[0:edge_start_index])
            last_end = edge_start_index + len(edges_str)
                
            # Create the graph and append it as the next segment
            graph = create_graph(ast.literal_eval(st.session_state['nodes']), edges)
            segments.append(graph)
            
            # Append text after the edges string as the last segment
            segments.append(response[last_end:])
            
            return segments

        else:
            return [response]

    except SyntaxError as e:
        print(e)
        print(response)

@st.cache_data
def create_graph(nodes, edges):
    """Create a graph from the nodes and edges."""
    g = net.Network(height="400px", width="100%", directed=True)
    g.repulsion(
        node_distance=320,
        central_gravity=0.33,
        spring_length=110,
        spring_strength=0.10,
        damping=0.95
    )
    for category, nodes_list in nodes.items():
        for node in nodes_list:
            g.add_node(node, label=node, color=category_colors.get(category, "white"))

    for edge in edges:
        g.add_edge(edge[0], edge[1])

    g.save_graph('graph.html')

    return 1
    


# def create_graph(categories, edges, width=None, height=None, dpi=100):
#     graph = graphviz.Digraph('G', format='png')
#     graph.attr(rankdir='LR', bgcolor='black')  # Set background color to black

#     if width and height:
#         graph.attr(size=f"{width},{height}!")
#         graph.attr(dpi=str(dpi))

#     # Apply color and styles to each node based on its category
#     for category, nodes in categories.items():
#         color = category_colors.get(category, "white")  # Default node color is white if category not found
#         fontcolor = "white" if color != "white" else "black"  # Set font color to white unless the node color is white
#         with graph.subgraph() as s:
#             s.attr('node', style='filled', fillcolor=color, fontcolor=fontcolor)
#             for node in nodes:
#                 s.node(node)

#     # Set edge attributes to be visible on a black background
#     graph.attr('edge', color='white')

#     # Add edges to the graph
#     for edge in edges:
#         graph.edge(edge[0], edge[1])

#     return graph


# @st.cache_data
# def create_graph(edges):
#     graph = graphviz.Digraph()
#     for edge in edges:
#         graph.edge(edge[0], edge[1])
#     return graph  # return the graph object instead of rendering it here




