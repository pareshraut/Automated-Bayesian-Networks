import streamlit as st
import re
from setup import Nodes, Edges, ProbabilityDistribution, Decision
from util import extract_and_format_nodes, extract_and_format_edges, get_edges, get_cpds, process_response, process_prob
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
import streamlit.components.v1 as components
from langchain import PromptTemplate, LLMChain

# openai_api_key = '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'


st.set_page_config(
    page_title="Bayesian - Risk Manager Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("styles.css") as f:                                                
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html= True) 


if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = None
    
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


# with open("risk_bot/styles.css") as f:                                                
#     st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html= True) 


header = st.container()
header.write(
    """
    <div class='fixed-header'>
        <h1 style="margin: 0;">Bayesian - Risk Manager</h1>
        <p style="margin: 5;">Helping you manage the risk in your scenario</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Check if 'openai_api_key' is already in the session state
if not st.session_state['openai_api_key']:
    # If not, use a sidebar for input
    with st.sidebar:
        key = st.text_input("OpenAI API Key", type="password")

    # Assign the key to the session state
    if key:
        st.session_state['openai_api_key'] = key

# After assigning, check if the API key is present in the session state
if not st.session_state.get('openai_api_key'):
    st.error("Please add your OpenAI API key in the sidebar.")
    st.stop()

# Initialize the session_state variables
if 'nodes' not in st.session_state:
    st.session_state['nodes'] = None
if 'edges' not in st.session_state:
    st.session_state['edges'] = None
if 'probability' not in st.session_state:
    st.session_state['probability'] = None
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi, Please tell me your scenario"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=50, return_messages=True)
if 'category' not in st.session_state:
    st.session_state['category'] = 'nodes'
if 'pattern' not in st.session_state:
    st.session_state['pattern'] = r"(?i)final nodes\s*:\s*(.*(?:\n|.)*)"
if 'sidebar_node' not in st.session_state:
    st.session_state['sidebar_node'] = None
if 'sidebar_edge' not in st.session_state:
    st.session_state['sidebar_edge'] = None
if 'sidebar_cpd' not in st.session_state:
    st.session_state['sidebar_cpd'] = None
if 'tentative_edges' not in st.session_state:
    st.session_state['tentative_edges'] = None
if 'tentative_cpds' not in st.session_state:
    st.session_state['tentative_cpds'] = None
    

class RiskBot:
    def __init__(self, api_key, memory):
        self.api_key = api_key
        self.chat = ChatOpenAI(temperature=0, openai_api_key=self.api_key, model='gpt-4-1106-preview', model_kwargs={'top_p':0.6})
        
        nodes_template = (
            "As a supportive AI, your objective is to assist the user in finalizing a list of nodes for a specified risk scenario "
            "within a Bayesian network framework. Initially, the user will share a scenario, and you are to respond with a list of nodes. "
            "Please categorize the nodes into the subsequent categories: "
            "Risk Node, Trigger Node, Mitigator Node and, Outcome Node (with exactly two nodes for Trigger Node and exactly one node for every other category)."
            "Ensure that each node is quantifiable so that data can be retrieved for analysis. "
            "Present each category as key and its corresponding nodes as value in JSON format, separated by lines. "
            'For instance - {'
            '"Risk Node": ["Oil price decreases"],'
            '"Trigger Node": ["Increase in Oil Production", "US GDP Growth"],'
            '"Mitigator Node": ["Decision: Buy/Sell/Hold"],'
            '"Outcome Node": ["Financial Result: Profit or Loss"]}'
            "ATTENTION: After presenting the suggested nodes, ask the user for feedback or modifications."
            "If the user provides feedback or requests changes to the nodes, incorporate their input and adjust the nodes accordingly. "
            "Ask the user if they are satisfied with the proposed nodes and if they wish to finalize the nodes for the Bayesian Network. "
            "If the user is not yet satisfied and requests further adjustments, iteratively provide revised nodes suggestions based on their feedback until they are ready to finalize the nodes. "
            "ATTENTION: ONCE THE USER REPLIES that he is fine with the nodes, I want you to give output in the below format: "
            'Final nodes: {'
            '"Risk Node": ["Oil price decreases"],'
            '"Opportunity Node": ["Oil price increases"],'
            '"Trigger Node": ["Increase in Oil Production", "US GDP Growth"],'
            '"Mitigator Node": ["Decision: Buy/Sell/Hold"],'
            '"Outcome Node": ["Financial Result: Profit or Loss"]}'
            "- enter the nodes that the user finalized and is satisfied with here in the format of the example given."
        )

        edges_template = (
            "As an AI specializing in Bayesian Networks, your task is to suggest potential edges "
            "for a given set of nodes. The network consists of nodes categorized as Risk Node, "
            "Trigger Nodes, Mitigator Node, and Outcome Node.\n\n"
            "Edge Connection Rules:\n"
            "- Connect Trigger Nodes to the Risk Node.\n"
            "- Connect Risk Node to both Mitigator Node and Outcome Node.\n"
            "- Connect Mitigator Node directly to Outcome Node.\n\n"
            "These rules are not provided by the user. You should directly provide the edges and not mention the rules in your response."
            "Ensure the network remains coherent, connected, and loop-free. Introduce intermediary nodes if needed to reduce complexity.\n\n"
            "Edges Representation:\n"
            "- Return edges as tuples in a list, for example:\n"
            "[('Increase in Oil Production', 'Oil price decreases'), ('US GDP Growth', 'Oil price increases'), "
            "('Oil price decreases', 'Decision: Buy/Sell/Hold'), ('Decision: Buy/Sell/Hold', 'Financial Result: Profit or Loss')]\n\n"
            "After suggesting edges, ask the user if modifications are needed and integrate feedback.\n\n"
            "Finalization:\n"
            "- Present the final list of edges once the user approves, updating as necessary to maintain network integrity as follows:\n"
            " Final edges: [('Increase in Oil Production', 'Oil price decreases'), ('US GDP Growth', 'Oil price increases'), "
            "('Oil price decreases', 'Decision: Buy/Sell/Hold'), ('Oil price decreases', 'Financial Result: Profit or Loss'), "
            "('Decision: Buy/Sell/Hold', 'Financial Result: Profit or Loss')].\n"
            "Remember to engage the user and maintain a coherent dialogue throughout the process."
        )

        probability_template = (
            "You are a part of an AI chatbot that is trying to help a user assess the risk of a trade using a Bayesian network. Your specific job is to help the user complete a Conditional Probability Distribution (CPD) table. You will be suggesting probabilities for each entry based on available data or user input, adjusting as needed. The user has no understanding of Bayesian networks and probabilites and your language is to reflect that. Use simple terms while communicating with the user.\n"
            "Step 1: Present the CPDs as html tables one by one. Use class='styled-table' for table and use headers where appropriate. \n"
            "Present the independent CPDs first, followed by the dependent CPDs.\n"
            "***IMPORTANT***: One message should contain only one CPD\n"
            "Below these tables, explain these probabilities to a user in natural language using non-technical jargon. Use everyday analogies. For example, there's a chance that outcome A could happen, and outcome B might happen this often...\n"
            "Step 2: Incorporate the user's feedback and adjust the probabilities accordingly.\n"
            "Step 3: Follow this method for all variables for which the CPDs are passed from the backend, one by one.\n"
            "Step 4: For the variables for which CPDs are missing, elicit the CPDs from the user. Present and confirm one probability from one variable at a time in a user-friendly manner. Remember to keep in mind all parents and their combinations.\n"
            "***IMPORTANT:*** A given response should focus on only one probability value of the CPD table.\n"
            "Step 5: Once the values for a given variable are elicited, present it to the user in the form of html table as above and confirm.\n"
            "Step 6: Once all the CPDs are confirmed, announce 'Done with all the probability values'.\n"
            "Step 7: Ask the user which of category of the risk node is most likely to occur. For instance if risk node is 'Market Risk' and the categories are 'High', 'Medium' and 'Low', ask the user which of these is most likely to occur. \n"
            "Step 8 : Once done with all above steps, announce 'Wait as we compute the final decision'.\n"
            "REMEMBER: Account for all relationships while eliciting the CPDs. "
        )

        decision_template = (
            "You are a Financial Risk Analyst. You will be provided with inference results for each category of the mitigator node. Go through this and follow the steps:\n"
            "step 1: Present the results in a table format.Use class='styled-table' for table and use headers where appropriate.\n"
            "step 2: Based on the results, identify the category of the mitigator node that has the highest probability of the trade outcome Gain.\n"
            "step 3: Suggest the mitigator category that should be selected to maximize the probability of the trade outcome Gain.\n"
            "step 4: Explain your decision in non technical terms. Be sure to explain the risk involved."
        )

        self.memory = memory
        self.nodes_handler = Nodes(self.memory, self.get_prompt(nodes_template), self.chat)
        self.edges_handler = Edges(self.memory, self.get_prompt(edges_template), self.chat)
        self.probability_handler = ProbabilityDistribution(self.memory, self.get_prompt(probability_template), self.chat)
        self.decision_handler = Decision(self.memory, self.get_prompt(decision_template), self.chat)
        
        self.nodes_pattern = r"(?i)final nodes\s*:\s*(.*(?:\n|.)*)"
        self.edges_pattern = r"(?i)final edges\s*:\s*(.*(?:\n|.)*)"
        self.probability_pattern = r"(?i)Wait as we compute the final decision"
        self.NODES = 'nodes'
        self.EDGES = 'edges'
        self.PROBABILITY = 'probability'
        self.DECISION = 'decision'


    def get_prompt(self, template):
        system_message_prompt = SystemMessagePromptTemplate.from_template(template, template_format="jinja2")  
        human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
        
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, MessagesPlaceholder(variable_name="history"), human_message_prompt]
        )
        return chat_prompt

    def get_handler(self):
        if st.session_state.category == self.NODES:
            return self.nodes_handler
        elif st.session_state.category == self.EDGES:
            return self.edges_handler
        elif st.session_state.category == self.PROBABILITY:
            return self.probability_handler
        else:
            return self.decision_handler

    def manage_risk(self, input_text):
        handler = self.get_handler()
        response = handler.get_response(input_text)
        if st.session_state.category == self.DECISION:
            return response
        else:
            match = re.search(st.session_state.pattern, response)
            if match:
                if st.session_state.category == self.NODES:
                    st.session_state.category = self.EDGES
                    st.session_state.pattern = self.edges_pattern
                    start_index = match.group(1).find('{')
                    end_index = match.group(1).find('}')
                    st.session_state.nodes = match.group(1)[start_index:end_index+1]
                    response = None
                elif st.session_state.category == self.EDGES:
                    st.session_state.category = self.PROBABILITY
                    st.session_state.pattern = self.probability_pattern
                    start_index = match.group(1).find('[')
                    end_index = match.group(1).find(']')
                    st.session_state.edges = match.group(1)[start_index:end_index+1]
                    response = None
                elif st.session_state.category == self.PROBABILITY:
                    st.session_state.category = self.DECISION
                    table = process_prob()
                    response = self.decision_handler.get_response(table)
            st.cache(allow_output_mutation=True)
            return response

risk_bot = RiskBot(st.session_state.openai_api_key, st.session_state.buffer_memory)


response_container = st.container()
text_container = st.container()

with text_container:
    query = st.chat_input("Query", key="input")
    if query:   
        with st.spinner("Thinking..."):
            response = risk_bot.manage_risk('User: ' + query)
            st.session_state['requests'].append(query)
            if response:
                st.session_state['responses'].append(response)

    # Update session_state according to risk_bot's internal state
    if st.session_state['nodes']:
        extract_and_format_nodes(st.session_state['nodes'])
        if not st.session_state['sidebar_node']:
            st.session_state['sidebar_node'] = True
            query = 'Backend: Give me the edges based on these nodes'
            response = risk_bot.edges_handler.get_response(query)
            st.session_state['responses'].append(response)
        # if not st.session_state['tentative_edges']:
        #     st.session_state['tentative_edges'] = get_edges(st.session_state['nodes'])
        #     query = 'Following are edges collected from dynamically collected data. Incorporate them in your response.{}.\nThis query does not come from the user.'.format(st.session_state['tentative_edges'])
        #     response = risk_bot.edges_handler.get_response(query)
        #     st.session_state['responses'].append(response)
    if st.session_state['edges']:
        extract_and_format_edges(st.session_state['edges'])
        if not st.session_state['sidebar_edge']:
            st.session_state['sidebar_edge'] = True
            st.session_state['tentative_cpds'] = get_cpds(st.session_state['edges'])
            query = 'Backend: Following are CPDs collected from dynamically collected data. These come from the backend and, not from the user. Incorporate them in your response.{}\n'.format(st.session_state['tentative_cpds'])
            response = risk_bot.probability_handler.get_response(query)
            st.session_state['responses'].append(response)
    if st.session_state['probability']:pass

problematic_tokens = ['```json', '```json\n', '```', '```\n', '```dot', '```graphviz', 'plaintext\n', 'plaintext', '```plaintext', '```plaintext\n']

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            segments = process_response(st.session_state['responses'][i])
            for segment in segments:    
                with st.chat_message("assistant"):
                    if isinstance(segment, str):
                        for token in problematic_tokens:
                            if token in segment:
                                segment = segment.replace(token, '')
                        st.write(segment.rstrip(), unsafe_allow_html=True)
                    else:
                        HtmlFile = open(f'graph.html', 'r', encoding='utf-8')
                        components.html(HtmlFile.read(), height=435)
            if i < len(st.session_state['requests']):
                with st.chat_message("user"):
                    st.write(st.session_state['requests'][i])

        while i < len(st.session_state['requests']):
            with st.chat_message("user"):
                st.write(st.session_state['requests'][i])
            i += 1



