import re
import streamlit as st
import bnlearn as bn
import graphviz
import ast
import datetime
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


# Set the FRED API key
os.environ["FRED_API_KEY"] = "b73c11e70992c0501f8748360e192763"

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

    with st.sidebar.expander("Current Edges"):
        for item in formatted_data:
            st.write(item)
    
    with st.sidebar.expander("Current Network"):
        graph = graphviz.Digraph()
        for edge in edges:
            graph.edge(edge[0], edge[1])
        st.graphviz_chart(graph)

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
        "input": '["Consumer Sentiment", "Housing Market Trends"]',
        "output": '{"Consumer Sentiment": ("FRED", "UMCSENT"), "Housing Market Trends": ("YFinance", "HOMZ")}'
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
        ("system", "You are an AI tasked with selecting the most suitable API for each of the following nodes' requirements. "
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
    if isinstance(value, tuple) and len(value) == 2:
        api, ticker = value
        df = get_data(api, ticker)
        if api == 'FRED':
            df = pd.DataFrame(df, columns=[node_name])
        else:
            df = df.rename(columns={'Close': node_name})
            df = df[[node_name]]
        return df
    else:
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
    
    # Step 4: Continue with your current logic.
    label_map = {2: ['Low', 'High'],
                 3: ['Low', 'Medium', 'High']}

    for column in df.columns:
        num_unique = df[column].nunique()
        
        if num_unique < 3:

            df = df.drop(columns=[column])
        else:
            # We use min(3, num_unique-1) to ensure that we don't go beyond the number of labels we have defined
            bins = min(3, num_unique - 1)
            labels = label_map[bins]
            df[column] = pd.qcut(df[column], q=bins, labels=labels, duplicates='drop')

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
    print(edges)
    return edges

def get_cpds(edges):
    """Compute the CPDs given the edges and nodes."""
    print('Inside get_cpds')
    print('edges: ', edges)
    edges = ast.literal_eval(edges)
    df = get_data_from_nodes(edges)
    valid_edges = [edge for edge in edges if edge[0] in df.columns and edge[1] in df.columns]
    bn = BayesianNetwork(valid_edges)
    bn.fit(df, estimator=MaximumLikelihoodEstimator)
    cpd_strings = ''
    for node in df.columns:
        cpd = bn.get_cpds(node)
        cpd_strings += cpd_to_string(cpd)
        cpd_strings += 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    return cpd_strings








