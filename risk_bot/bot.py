import streamlit as st
import re
import graphviz
import ast
from setup import Nodes, Edges, ProbabilityDistribution
from util import extract_and_format_nodes, extract_and_format_edges, get_empty_cpd, get_edges, get_cpds
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
            'For instance - {"Event Node": ["Risk of Apple stock going down by more than 5% in a week"], '
            '"Opportunity Node": ["Percentage Change in Competitive Stock Price"], '
            '"Trigger Nodes": ["Percentage Change in S&P 500 Index", "Percentage Change in Nasdaq 100 Index"], '
            '"Mitigator Nodes": ["Apple\'s Dividend Yield", "Apple\'s Buyback Announcements"], '
            '"Control Nodes": ["Percentage of Portfolio in Cash", "Percentage of Portfolio in Bond Investments"], '
            '"External Influence Nodes": ["Federal Interest Rate", "US GDP Growth Rate"], '
            '"Outcome Nodes": ["Percentage Change in Portfolio Value", "Return on Investment (ROI)"}. '
            "ATTENTION:  After presenting the suggested nodes, ask the user : 'Do you have any feedback or modification for the nodes provided'"
            "If the user provides feedback or requests changes to the nodes, incorporate their input and adjust the nodes accordingly."
            "Ask the user if they are satisfied with the proposed nodes and if they wish to finalize the nodes for the Bayesian Network"
            "If the user is not yet satisfied and requests further adjustments, iteratively provide revised nodes suggestions based on their feedback until they are ready to finalize the nodes."
            """ATTENTION: ONCE THE USER REPLIES that he is fine with the nodes , I want you to give output in the below format:
            Final nodes : enter the nodes that he finalized and is satisfied with here in the format of the example given
            """
        )

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

        probability_template = (
            """
### Initial Instructions:
- Your goal is to assist in filling in the Conditional Probability Distribution (CPD) table provided by the user. You will suggest probabilities for each entry and adjust them based on the user's feedback.
  
- If the user indicates that the conversation is too technical, you should simplify your explanations.

### Visual Representation:
- To aid visualization, you might occasionally sketch out relationships between nodes, especially if it helps clarify the dependencies.
  
- For instance, if A and B are influencing C, you could visualize it as:
  A (Category) -> C (Category) <- B (Category)

- You can also use a CPD table to aid the user in visualizing the dependencies between nodes as well as categories of each node.

### Data Collection:
1. Starting with the CPD table for an independent node, such as [Node Name]:
   - "Looking at [Node Name], potential categories could be '[Category 1]', '[Category 2]'. Based on your understanding, the probability for [Category 1] might be [X%]. Does this align with the user's thinking or would they suggest a different value?"

2. For dependent nodes, where the probability of a node depends on the state of its parents:
   - "Now, considering [Dependent Node Name] which is influenced by [Parent Node(s) Name]. Given [Parent Node Category], the probability of [Dependent Node Name] being in [Category] might be [Y%]. How does the user feel about this value?"

3. You will proceed this way, filling out each entry of the CPD table one by one, proposing a value and adjusting it based on the user's feedback.

### Continuous Feedback:
- You should always be receptive to guidance from the user. If the user feels there's a more appropriate probability or if your suggestion doesn't seem right, you will adjust accordingly.

### Completion:
- After you've filled out the entire CPD table, you will provide a summary of the inputs and express gratitude for the user's collaboration.
""")

        # )
        # probability_template = (
        #     "Your overarching task is to methodically elicit and document the probability distributions for each node in the Bayesian Network, ensuring clarity and precision at each step. Here's a step-by-step breakdown:"
            
        #     " 1. **Node Identification**:"
        #     "    - Begin by methodically examining the network to identify the true independent nodes, which are nodes that do not have any incoming edges."
            
        #     " 2. **Category Suggestions and Confirmation**:"
        #     "    - For every node, whether independent or dependent, propose appropriate categorizations based on its context and semantic nature."
        #     "    - Interact with the user in a structured manner. For instance: 'For the node {Node Name}, we're considering the categories {Suggested Categories}. Do these resonate with your expectations, or would you prefer modifications?'"
        #     "    - Be open to feedback. Adjust and reconfirm categories based on the user's input until mutual agreement is achieved for each node."
            
        #     " 3. **Probability Elicitation for Independent Nodes**:"
        #     "    - Once all categories for the independent nodes are mutually agreed upon, initiate the probability elicitation phase."
        #     "    - Instead of overwhelming the user with multiple requests, methodically ask for a single probability value corresponding to each category of an independent node. For instance: 'Considering the node {Node Name}, what's your estimated probability for the category {Category Name}?'"
        #     "    - Take note of the user's response before proceeding to the next category or node."
            
        #     " 4. **Probability Elicitation for Dependent Nodes**:"
        #     "    - With the independent nodes' probabilities documented, shift your focus to the dependent nodes."
        #     "    - Given the categories of their parent nodes, solicit the conditional probabilities in a systematic manner, iterating through each combination of parent node categories."
        #     "    - Prompt the user with structured queries, like: 'Given that {Parent Node 1} is {Category A} and {Parent Node 2} is {Category B}, what's your estimated probability for {Dependent Node} being {Category X}?'"
            
        #     " 5. **Documentation and Recap**:"
        #     "    - As you accumulate these probabilities, ensure they're being systematically documented."
        #     "    - Once all probabilities are captured, present a structured overview to the user for final verification. Ask: 'Here's the compiled probability distribution based on our discussion. Do these values align with your expectations, or are there any final adjustments you'd like to make?'"

        #     "Remember, throughout this process, clarity, patience, and user feedback are paramount. The objective is not just to obtain probabilities but to ensure that the user is confident and clear about every value they provide."
        # )
        

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
                st.session_state['pattern'] = self.edges_pattern
                st.session_state['nodes'] = match.group(1)
            elif st.session_state.category == self.EDGES:
                st.session_state.category = self.PROBABILITY
                st.session_state['pattern'] = self.probability_pattern
                st.session_state['edges'] = match.group(1)
            else:
                st.session_state['probability'] = match.group(1)
        st.cache(allow_output_mutation=True)
        return response
        
@st.cache_data
def create_graph(edges):
    graph = graphviz.Digraph()
    for edge in edges:
        graph.edge(edge[0], edge[1])
    return graph  # return the graph object instead of rendering it here

def process_response(response):
    edge_pattern = re.compile(r"\[\s*((?:\([^)]+\)\s*,?\s*)+)\]")
    match = re.search(edge_pattern, response)
    segments = []
    last_end = 0
    edges = []
    if match:
        edges = ast.literal_eval(match.group(0))
    else:
        return [response]
    
    index = response.find(match.group(0))
    print(index)
    if index != -1:
        segments.append(response[0:index])
        last_end = index + len(match.group(0))

    # Create the graph and append it as the last segment
    graph = create_graph(edges)
    segments.append(graph)
    # Append text after the edges string
    segments.append(response[last_end:])

    # # Create the graph and append it as the last segment
    # graph = create_graph(edges)
    # segments.append(graph)

    return segments


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
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=20, return_messages=True)
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
        if not st.session_state['tentative_edges']:
            st.session_state['tentative_edges'] = get_edges(st.session_state['nodes'])
            query = 'I have edges denoting common causal relationships between nodes. You can utilize these or propose additional suggestions.{}'.format(st.session_state['tentative_edges'])
            response = risk_bot.edges_handler.get_response(query)
            st.session_state['responses'].append(response)
    if st.session_state['edges']:
        extract_and_format_edges(st.session_state['edges'])
        if not st.session_state['tentative_cpds']:
            st.session_state['tentative_cpds'] = get_cpds(st.session_state['edges'])
            query = 'I have a CPD table denoting the probability distribution of each node given the state of its parents. You can utilize these or propose additional suggestions.{}'.format(st.session_state['tentative_cpds'])
            response = risk_bot.probability_handler.get_response(query)
            st.session_state['responses'].append(response)
    if st.session_state['probability']:pass

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            segments = process_response(st.session_state['responses'][i])
            displayed_graph = False
            for j, segment in enumerate(segments):
                if isinstance(segment, str) and segment.strip():  # Check if the segment is a non-empty string
                    if not displayed_graph:
                        # First, check if this is the graph segment
                        if "graph" in segment.lower():  # Adjust as needed
                            st.graphviz_chart(segment)
                            displayed_graph = True
                        else:
                            message(segment, key=f"{i}_text_{j}", avatar_style="bottts-neutral", seed = 'Aneka')
                    else:
                        message(segment, key=f"{i}_text_{j}", avatar_style="bottts-neutral",seed = 'Aneka')
                else:
                    if not displayed_graph:
                        st.graphviz_chart(segment)
                        displayed_graph = True
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=f"{i}_user", avatar_style= "bottts-neutral", seed = 'Molly')



