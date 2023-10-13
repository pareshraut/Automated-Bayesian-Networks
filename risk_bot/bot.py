import streamlit as st
import re
from setup import Nodes, Edges, ProbabilityDistribution
from util import extract_and_format_nodes, extract_and_format_edges
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from streamlit_chat import message
from langchain.memory import ConversationBufferWindowMemory

openai_api_key = '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'

st.title("Bayesian - Risk Manager")
st.write("""
I am a Bayesian Risk Manager Bot, I can help you manage the risk in your scenario.
""")

class RiskBot:
    def __init__(self, api_key, memory):
        self.api_key = api_key
        self.chat = ChatOpenAI(temperature=0, openai_api_key=self.api_key, model='gpt-4')
        
        nodes_template = (
            "As a supportive AI, your objective is to assist the user in finalizing a list of nodes for a specified risk scenario "
            "within a Bayesian network framework. Initially, the user will share a scenario, and you are to respond with a list of nodes. "
            "Please categorize the nodes into the subsequent categories: "
            "Event Node, Opportunity Node (one node each), Trigger Nodes, "
            "Mitigator Nodes, Control Nodes, External Influence Nodes, and Outcome Nodes (multiple nodes allowed for these categories). "
            "Present each category and its corresponding nodes in JSON format, separated by lines. "
            'For instance - {"Event Node": ["Stock Market Crash"], "Opportunity Node": ["Investment Opportunity in Tech Stocks"], '
            '"Trigger Nodes": ["Economic Recession", "High Unemployment Rate"], '
            '"Mitigator Nodes": ["Government Intervention", "Monetary Policy"], '
            '"Control Nodes": ["Investment in Safe Assets", "Diversification Strategy"], '
            '"External Influence Nodes": ["Global Political Instability"], '
            '"Outcome Nodes": ["Investment Loss", "Investment Gain"]}. '
            "ATTENTION:  After presenting the suggested nodes, ask the user : 'Do you have any feedback or modification for the nodes provided'"
            "If the user provides feedback or requests changes to the nodes, incorporate their input and adjust the nodes accordingly."
            "Ask the user if they are satisfied with the proposed nodes and if they wish to finalize the nodes for the Bayesian Network"
            "If the user is not yet satisfied and requests further adjustments, iteratively provide revised nodes suggestions based on their feedback until they are ready to finalize the nodes."
            """ATTENTION: ONCE THE USER REPLIES that he is fine with the nodes , I want you to give output in the below format:
            Final nodes : enter the nodes that he finalized and is satisfied with here in the format of the example given
            """
        )

        # edges_template =("Now that the nodes are finalized, you are to assist the user in finalizing the edges for the Bayesian network. ")

        edges_template = (
            "In the role of a supportive AI, your objective is to assist the user in finalizing a list of edges for a specified risk scenario within a Bayesian network framework. "
            "You will adopt a 'Tree of Thought' reasoning strategy, systematically examining possible edge configurations while ensuring the integrity of the Bayesian Network by avoiding loops or disconnections. "
            "Starting with the user-mentioned scenario: 'Apple Stock going down by 5 percent in the next week,' we will identify the pre-specified nodes and explore potential causal relationships between them. "
            "We may engage in a collaborative effort with a panel of experts, iterating through edge definitions and configurations, refining the edges based on collective insights until a consensus on a valid edge configuration is reached. "
            "Present the proposed edges to the user in a structured manner, illustrating the causal relationships between nodes. "
            "Each edge is represented as a tuple of two nodes, where the first node is the parent node and the second node is the child node. "
            "For instance - [(\"Market Conditions\", \"Apple Stock\")] "
            "Return the edges as a list of tuples. Here is an example: "
            """[(\"Market Conditions\", \"Apple Stock\"), (\"Investor Sentiment\", \"Apple Stock\"),
            (\"Apple Stock\", \"Apple Stock Price\"), (\"Apple Stock Price\", \"Apple Stock Return\")], 
            (\"Apple Stock Return\", \"Investment Return\")]"""
            "After presenting the suggested edges, prompt the user with the question: 'Are you satisfied with the provided edges, or would you like to modify some of them?' "
            "If the user requests changes to the edges, incorporate their input and adjust the edges accordingly, ensuring no closed loops or disconnected nodes. "
            "If any closed loops or disconnected nodes occur, inform the user that the provided edges are not valid and ask them to revise. "
            "Display the updated set of edges and prompt the user again: 'Are you satisfied with the updated edges, or would you like to further modify them?' "
            """ATTENTION: ONCE THE USER REPLIES that he is fine with the edges , I want you to give output in the below format:
            Final edges : enter the edges that he finalized and is satisfied with here in the format of the example given
            """
        )
        probability_template = ("Now that the nodes and edges are finalized, you are to assist the user in finalizing the probability distribution for the Bayesian network.")

        #self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        self.memory = memory
        self.nodes_handler = Nodes(self.memory, self.get_prompt(nodes_template), self.chat)
        self.edges_handler = Edges(self.memory, self.get_prompt(edges_template), self.chat)
        self.probability_handler = ProbabilityDistribution(self.memory, self.get_prompt(probability_template), self.chat)
        self.nodes_pattern = r"(?i)final nodes\s*:\s*(.*(?:\n|.)*)"
        self.edges_pattern = r"(?i)final edges\s*:\s*(.*(?:\n|.)*)"
        self.probability_pattern = r"(?i)final probability distribution\s*:\s*(.*(?:\n|.)*)"
        self.NODES = 'nodes'
        self.EDGES = 'edges'
        self.PROBABILITY = 'probability'


    def get_prompt(self, template):
        system_message_prompt = SystemMessagePromptTemplate.from_template(template, template_format="jinja2")  
        human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
        
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, MessagesPlaceholder(variable_name="history"), human_message_prompt]
        )
        return chat_prompt

    def get_current_handler_and_pattern(self):
        if st.session_state.category == self.NODES:
            return self.nodes_handler
        elif st.session_state.category == self.EDGES:
            return self.edges_handler
        else:
            return self.probability_handler

    def manage_risk(self, input_text):
        handler = self.get_current_handler_and_pattern()
        response = handler.get_response(input_text)
        print('Pattern: ', st.session_state['pattern'])
        match = re.search(st.session_state['pattern'], response)
        if match:
            # print match
            print('match: ', match)
            if st.session_state.category == self.NODES:
                st.session_state.category = self.EDGES
                st.session_state['pattern'] = self.edges_pattern
                response = self.edges_handler.get_response('Give me the edges now.')
                st.session_state['nodes'] = match.group(1)
            elif st.session_state.category == self.EDGES:
                st.session_state.category = self.PROBABILITY
                st.session_state['pattern'] = self.probability_pattern
                response = self.probability_handler.get_response('Give me the probability distribution now.')
                st.session_state['edges'] = match.group(1)
            else:
                st.session_state['probability'] = match.group(1)
        st.cache(allow_output_mutation=True)
        return response
        
    

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
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=10, return_messages=True)
if 'category' not in st.session_state:
    st.session_state.category = 'nodes'
if 'pattern' not in st.session_state:
    st.session_state.pattern = r"(?i)final nodes\s*:\s*(.*(?:\n|.)*)"

risk_bot = RiskBot(openai_api_key, st.session_state.buffer_memory)


response_container = st.container()
text_container = st.container()


with text_container:
    query = st.chat_input("Query", key="input")
    if query:    
        with st.spinner("Thinking..."):
            response = risk_bot.manage_risk(query)
            st.session_state['requests'].append(query)
            st.session_state['responses'].append(response)

    # Update session_state according to risk_bot's internal state
    if st.session_state['nodes']:
        extract_and_format_nodes(st.session_state['nodes'])
    if st.session_state['edges']:
        extract_and_format_edges(st.session_state['edges'])
    if st.session_state['probability']:pass

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')


