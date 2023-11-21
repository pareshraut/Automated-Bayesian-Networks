import streamlit as st
import re
from setup import Nodes, Edges, ProbabilityDistribution
from util import extract_and_format_nodes, extract_and_format_edges, get_edges, get_cpds, process_response
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from streamlit_chat import message
from langchain.memory import ConversationBufferWindowMemory
import streamlit.components.v1 as components

# openai_api_key = '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'

st.set_page_config(
    page_title="Bayesian - Risk Manager Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",

)

with open("styles.css") as f:                                                
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html= True) 

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

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

# st.title("Bayesian - Risk Manager")
# st.write("""
# I am a Bayesian Risk Manager Bot, helping you manage the risk in your scenario.
# """)


if not openai_api_key:
    st.error("Please add your OpenAI API key in the sidebar")
    st.stop()

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


#         probability_template = (
#     """### Initial Instructions:
# - Your role is to aid the user in completing a Conditional Probability Distribution (CPD) table. Suggest probabilities for each entry based on available data or user input, adjusting as needed.
  
# - Communicate in straightforward terms to ensure understanding, particularly if the user is unfamiliar with technical concepts.

# ### Visual Representation:
# - Illustrate dependencies between nodes when helpful, using a format like:
#   A (Trigger Node) -> C (Risk Node) <- B (Control Node)

# - Display CPD tables with missing values as '?', enclosed in triple backticks.

# ### Data Collection:
# 1. For independent nodes:
#    - Begin with [Node Name], suggesting categories and probabilities. Confirm or revise these with the user.

# 2. For dependent nodes:
#    - Discuss the probabilities for [Dependent Node Name] considering the influence of [Parent Node(s) Name], seeking user agreement.

# 3. Fill out the CPD table systematically, addressing one entry at a time and adapting to user feedback.

# ### Continuous Feedback:
# - Be responsive to user guidance, ready to modify probability estimates according to their suggestions.

# ### Completion:
# - After completing the CPD table, provide a summary and thank the user for their collaboration.

# Note: Replace '?' with the suggested probabilities or leave it as a placeholder for the user to fill in. Use this as a template for the visual representation and not for the structure inside..
# """
# )

        probability_template = (
            "### Step-by-Step Iterative Process for CPD Confirmation:\n\n"
            "Step 1: Introduction\n"
            "- Begin by explaining to the user that you will assist in completing a Conditional Probability Distribution (CPD) table, using easy-to-understand language.\n"
            "- Clarify that the process is designed for those without a background in probability.\n\n"
            "Step 2: Initial CPD Presentation in Natural Language\n"
            "- Present the first CPD derived from data in simple, natural language.\n"
            "- Example presentation: 'For the first variable, let's consider how likely different outcomes are. For example, there's a chance that outcome A could happen, and outcome B might happen this often...'\n"
            "- Encourage the user to provide their thoughts and feedback on these initial probability estimations.\n\n"
            "Step 3: User Feedback and Adjustment\n"
            "- Discuss the presented probabilities for the variable with the user, using everyday terms.\n"
            "- Adjust the probabilities based on user feedback, still using natural language for clarity.\n\n"
            "Step 4: Confirm and Finalize CPD for One Variable\n"
            "- Once the user confirms the probabilities for the first variable, prepare the final CPD in the specified dictionary format for backend parsing.\n"
            "- This step will not be presented to the user but is rather for backend processing.\n"
            "- Example final CPD format:\n"
            "  ```\n"
            "  final_cpd_dict = {\n"
            "      'variable': 'Confirmed Variable',\n"
            "      'variable_cardinality': Number of States,\n"
            "      'probabilities': {'State 1': Confirmed Probability, 'State 2': Confirmed Probability},\n"
            "      'evidence': ['Parent Node 1'],\n"
            "      'evidence_cardinality': [Cardinality],\n"
            "      'state_names': {'Variable Name': ['State 1', 'State 2']}\n"
            "  }\n"
            "  ```\n\n"
            "Step 5: Iterative Process for Subsequent Variables\n"
            "- Repeat Steps 2 to 4 for each subsequent variable. Present and confirm one variable at a time in a user-friendly manner.\n"
            "- Ensure each variable is fully confirmed and finalized before moving to the next.\n\n"
            "Step 6: Completion of All Variables\n"
            "- After all variables have been discussed, adjusted, and confirmed, announce 'Done with all the probability values'.\n\n"
            "### Gradual Elicitation and Confirmation:\n"
            "- Throughout the process, explain probabilities and outcomes in everyday language, avoiding technical terms.\n"
            "- Focus on understanding the user's perspective and clarify any doubts in a friendly manner.\n\n"
            "REMEMBER: Present initial CPDs in natural language for user understanding. Return confirmed CPDs in dictionary format for backend parsing, focusing on one variable at a time."
        )

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
        match = re.search(st.session_state['pattern'], response)
        if match:
            if st.session_state.category == self.NODES:
                st.session_state.category = self.EDGES
                st.session_state.pattern = self.edges_pattern
                start_index = match.group(1).find('{')
                end_index = match.group(1).find('}')
                st.session_state.nodes = match.group(1)[start_index:end_index+1]
            elif st.session_state.category == self.EDGES:
                st.session_state.category = self.PROBABILITY
                st.session_state.pattern = self.probability_pattern
                start_index = match.group(1).find('[')
                end_index = match.group(1).find(']')
                st.session_state.edges = match.group(1)[start_index:end_index+1]
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
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=50, return_messages=True)
if 'category' not in st.session_state:
    st.session_state['category'] = 'nodes'
if 'pattern' not in st.session_state:
    st.session_state['pattern'] = r"(?i)final nodes\s*:\s*(.*(?:\n|.)*)"
if 'tentative_edges' not in st.session_state:
    st.session_state['tentative_edges'] = None
if 'tentative_cpds' not in st.session_state:
    st.session_state['tentative_cpds'] = None
    

risk_bot = RiskBot(openai_api_key, st.session_state.buffer_memory)


response_container = st.container()
text_container = st.container()

with text_container:

    query = st.chat_input("Query", key="input")
    if query:   
        with st.spinner("Thinking..."):
            response = risk_bot.manage_risk(query)
            st.session_state['requests'].append(query)
            if not re.search(risk_bot.nodes_pattern, response) and not re.search(risk_bot.edges_pattern, response) and not re.search(risk_bot.probability_pattern, response):
                st.session_state['responses'].append(response)

    # Update session_state according to risk_bot's internal state
    if st.session_state['nodes']:
        extract_and_format_nodes(st.session_state['nodes'])
        query = 'Give me the edges based on these nodes'
        response = risk_bot.edges_handler.get_response(query)
        st.session_state['responses'].append(response)
        # if not st.session_state['tentative_edges']:
        #     st.session_state['tentative_edges'] = get_edges(st.session_state['nodes'])
        #     query = 'Following are edges collected from dynamically collected data. Incorporate them in your response.{}.\nThis query does not come from the user.'.format(st.session_state['tentative_edges'])
        #     response = risk_bot.edges_handler.get_response(query)
        #     st.session_state['responses'].append(response)
    if st.session_state['edges']:
        extract_and_format_edges(st.session_state['edges'])
        if not st.session_state['tentative_cpds']:
            st.session_state['tentative_cpds'] = get_cpds(st.session_state['edges'])
            query = 'Following are CPDs collected from dynamically collected data. Incorporate them in your response.{}\nThis query does not come from the user.'.format(st.session_state['tentative_cpds'])
            response = risk_bot.probability_handler.get_response(query)
            st.session_state['responses'].append(response)
    if st.session_state['probability']:pass

problematic_tokens = ['```json', '```json\n', '```', '```\n', '```dot', '```graphviz', 'plaintext\n', 'plaintext', '```plaintext', '```plaintext\n']

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            segments = process_response(st.session_state['responses'][i])
            #displayed_graph = False
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
                    # else:
                    #     if not displayed_graph:
                    #         st.graphviz_chart(segment)
                    #         displayed_graph = True
            if i < len(st.session_state['requests']):
                with st.chat_message("user"):
                    st.write(st.session_state['requests'][i])
                #message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')



