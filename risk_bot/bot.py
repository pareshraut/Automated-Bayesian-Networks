import streamlit as st
import re
import graphviz
import ast
import json
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
        
        # nodes_template = (
        #     "As a supportive AI, your objective is to assist the user in finalizing a list of nodes for a specified risk scenario "
        #     "within a Bayesian network framework. Initially, the user will share a scenario, and you are to respond with a list of nodes. "
        #     "Please categorize the nodes into the subsequent categories: "
        #     "Event Node, Opportunity Node (one node each), Trigger Nodes, "
        #     "Mitigator Nodes, Control Nodes, External Influence Nodes, and Outcome Nodes (multiple nodes allowed for these categories). "
        #     "Present each category and its corresponding nodes in JSON format, separated by lines. "
        #     'For instance - {"Event Node": ["Risk of Apple stock going down by more than 5% in a week"], '
        #     '"Opportunity Node": ["Percentage Change in Competitive Stock Price"], '
        #     '"Trigger Nodes": ["Percentage Change in S&P 500 Index", "Percentage Change in Nasdaq 100 Index"], '
        #     '"Mitigator Nodes": ["Apple\'s Dividend Yield", "Apple\'s Buyback Announcements"], '
        #     '"Control Nodes": ["Percentage of Portfolio in Cash", "Percentage of Portfolio in Bond Investments"], '
        #     '"External Influence Nodes": ["Federal Interest Rate", "US GDP Growth Rate"], '
        #     '"Outcome Nodes": ["Percentage Change in Portfolio Value", "Return on Investment (ROI)"}. '
        #     "ATTENTION:  After presenting the suggested nodes, ask the user : 'Do you have any feedback or modification for the nodes provided'"
        #     "If the user provides feedback or requests changes to the nodes, incorporate their input and adjust the nodes accordingly."
        #     "Ask the user if they are satisfied with the proposed nodes and if they wish to finalize the nodes for the Bayesian Network"
        #     "If the user is not yet satisfied and requests further adjustments, iteratively provide revised nodes suggestions based on their feedback until they are ready to finalize the nodes."
        #     """ATTENTION: ONCE THE USER REPLIES that he is fine with the nodes , I want you to give output in the below format:
        #     Final nodes : enter the nodes that he finalized and is satisfied with here in the format of the example given
        #     """
        # )
        nodes_template = (
    "As a supportive AI, your objective is to assist the user in finalizing a list of nodes for a specified risk scenario "
    "within a Bayesian network framework. Initially, the user will share a scenario, and you are to respond with a list of nodes. "
    "Please categorize the nodes into the subsequent categories: "
    "Event Node, Opportunity Node (one node each), Trigger Nodes, "
    "Mitigator Nodes, Control Nodes, External Influence Nodes, and Outcome Nodes (with exactly one risk outcome and one reward outcome). "
    "Ensure that each node is quantifiable so that data can be retrieved for analysis. "
    "Present each category and its corresponding nodes in JSON format, separated by lines. "
    'For instance - {"Event Node": ["CPI Increase"], '
    '"Opportunity Node": ["Core CPI Stability"], '
    '"Trigger Nodes": ["Global oil price trends", "Import/export price indices"], '
    '"Mitigator Nodes": ["Federal interest rate hike", "Decrease in M2 money supply"], '
    '"Control Nodes": ["Fiscal policy changes"], '
    '"External Influence Nodes": ["Global economic events"], '
    '"Outcome Nodes": ["Hyperinflation scenario", "Inflation within target range"]}. '
    "ATTENTION: After presenting the suggested nodes, ask the user : 'Do you have any feedback or modification for the nodes provided?' "
    "If the user provides feedback or requests changes to the nodes, incorporate their input and adjust the nodes accordingly. "
    "Ask the user if they are satisfied with the proposed nodes and if they wish to finalize the nodes for the Bayesian Network. "
    "If the user is not yet satisfied and requests further adjustments, iteratively provide revised nodes suggestions based on their feedback until they are ready to finalize the nodes. "
    "ATTENTION: ONCE THE USER REPLIES that he is fine with the nodes, I want you to give output in the below format: "
    'Final nodes: {"Event Node": ["CPI Increase"], '
    '"Opportunity Node": ["Core CPI Stability"], '
    '"Trigger Nodes": ["Global oil price trends", "Import/export price indices"], '
    '"Mitigator Nodes": ["Federal interest rate hike", "Decrease in M2 money supply"], '
    '"Control Nodes": ["Fiscal policy changes"], '
    '"External Influence Nodes": ["Global economic events"], '
    '"Outcome Nodes": ["Hyperinflation scenario", ="Inflation within target range"]} '
    "- enter the nodes that the user finalized and is satisfied with here in the format of the example given."
)



        # edges_template = (
        #     "As an AI developed to assist in risk analysis within a Bayesian network, your role is to help construct a coherent list of edges "
        #     "reflecting the scenario: 'Apple Stock going down by 5 percent in the next week'. "
        #     "The edges should align with predefined nodes and the structure dictated by data-driven analysis. "
        #     "The established rules for forming these causal connections are: "
        #     "1. Connect each 'Trigger Node' to the 'Event Node'. "
        #     "2. Connect each 'Trigger Node' to the 'Opportunity Node'. "
        #     "3. Link 'Mitigator Nodes' to at least one 'Outcome Node'. "
        #     "4. Link both the 'Event Node' and 'Opportunity Node' to at least one 'Outcome Node'. "
        #     "5. Link each 'External Influence Node' to one of the 'Outcome Nodes'. "
        #     "6. Link each 'Control Node' to one of the 'Outcome Nodes'. "
        #     "**Make sure there are no loops or disconnectivity in the grpah**"
        #     "**The provided suggested edges come from the data collected dynamically based on the nodes and not from the user. So your response should reflect that.**"
        #     "**If these suggested edges deviate from the rules above, it should be incorporated but at the same time informed to the user.**"
        #     "I want you to understand that the first suggestion for the edges come from dynamic data and not from the user. Your language while addressing the user should reflect that."
        #     "**Always present edges as tuples in a list**. For example: "
        #     "[('Market Conditions', 'Apple Stock'), ('Investor Sentiment', 'Apple Stock'), "
        #     "('Apple Stock', 'Apple Stock Price'), ('Apple Stock Price', 'Apple Stock Return'), "
        #     "('Apple Stock Return', 'Investment Return')]"
        #     "Ensure that the list of tuples holds the norm for a python list of tuples. "
        #     "After presenting the edges and any deviations, ask the user: 'Are you satisfied with the provided edges, or would you like to modify some of them?' "
        #     "Incorporate user feedback to refine the edges, ensuring the network remains valid and well-structured. "
        #     "Once the user is content with the edge configuration, present the final list as: "
        #     "Final edges: [('Parent Node', 'Child Node'), ...]. "
        #     "If further adjustments are needed, update the edge list accordingly and re-prompt the user, always striving for a coherent and loop-free network."
        # )

        edges_template = (
            "You are an AI that specializes in constructing and refining the structure of Bayesian Networks. "
            "Your role is to suggest potential edges for a given set of nodes and to help incorporate user feedback "
            "to optimize the network's configuration. The network consists of nodes categorized as: Event Node, "
            "Opportunity Node, Trigger Nodes, Mitigator Nodes, Control Nodes, Impediment Nodes, External Influence Nodes, Outcome Nodes.\n\n"
            "**Your tasks are as follows:**\n\n"
            "1. **Edge Connection Rules:**\n"
            "   - Connect the Event Node to the Risk Outcome.\n"
            "   - Connect the Opportunity Node to the Reward Outcome.\n"
            "   - Connect Trigger Nodes to both the Event Node and the Opportunity Node.\n"
            "   - Connect Mitigator Nodes to the Risk Outcome.\n"
            "   - Connect Control Nodes to the Event Node.\n"
            "   - Connect External Influence Nodes to one of the Outcome Nodes.\n"
            "   - Connect Impediment Nodes to the Opportunity Node and the Reward Outcome.\n\n"
            "2. **Complexity Management:**\n"
            "   - Introduce an intermediary node when any node's connections exceed three, to maintain manageable complexity.\n\n"
            "3. **Network Consistency:**\n"
            "   - Ensure that the network remains coherent, connected and loop-free. Add or remove an edge if that is not the case and apprise the user of the same.\n\n"
            "4. **Suggestions and Feedback Integration:**\n"
            "   - You will receive a list of edge suggestions based on dynamic data. Review and suggest necessary additions "
            "or modifications to align with the established connection rules, noting deviations prompted by data-driven recommendations.\n\n"
            "**Response Format:**\n\n"
            "- Present all proposed edges as a list of tuples, following Python's list and tuple syntax. For example:\n"
            "  [\n"
            "    ('Fiscal policy changes', 'CPI Increase'),\n"
            "    ('Core CPI Stability', 'Inflation within target range'),\n"
            "    ('CPI Increase', 'Hyperinflation scenario'),\n"
            "    ('Federal interest rate hike', 'Hyperinflation scenario'),\n"
            "    ('Decrease in M2 money supply', 'Hyperinflation scenario'),\n"
            "    ('Global oil price trends', 'CPI Increase'),\n"
            "    ('Import/export price indices', 'Core CPI Stability'),\n"
            "    ...\n"
            "  ]\n"
            "- After presenting the suggested edges, including any deviations, ask the user:\n"
            "  \"Are you satisfied with the provided edges, or would you like to modify some of them?\"\n"
            "- Integrate user feedback to refine the edges while maintaining a valid and well-structured network.\n\n"
            "**Finalization:**\n\n"
            "- Once the user approves the edge configuration, present the final list as:\n"
            "  Final edges: [('Fiscal policy changes', 'CPI Increase'), ('Core CPI Stability', 'Inflation within target range'), ('CPI Increase', 'Hyperinflation scenario'), ('Federal interest rate hike', 'Hyperinflation scenario'), ('Decrease in M2 money supply', 'Hyperinflation scenario'), ('Global oil price trends', 'CPI Increase'), ('Import/export price indices', 'Core CPI Stability')].\n"
            "- Should further adjustments be required, update the edge list accordingly and re-prompt the user, "
            "always ensuring the network is coherent and free of loops.\n\n"
            "**Remember:**\n"
            "- Maintain a clear, concise, and coherent dialogue.\n"
            "- Uphold the structural integrity of the Bayesian Network throughout the interaction.\n"
            "- Keep the user engaged and informed during the edge refinement process."
        )


        probability_template = (
    """### Initial Instructions:
- Your role is to aid the user in completing a Conditional Probability Distribution (CPD) table. Suggest probabilities for each entry based on available data or user input, adjusting as needed.
  
- Communicate in straightforward terms to ensure understanding, particularly if the user is unfamiliar with technical concepts.

### Visual Representation:
- Illustrate dependencies between nodes when helpful, using a format like:
  A (Trigger Node) -> C (Risk Node) <- B (Control Node)

- Display CPD tables with missing values as '?', enclosed in triple backticks.

### Data Collection:
1. For independent nodes:
   - Begin with [Node Name], suggesting categories and probabilities. Confirm or revise these with the user.

2. For dependent nodes:
   - Discuss the probabilities for [Dependent Node Name] considering the influence of [Parent Node(s) Name], seeking user agreement.

3. Fill out the CPD table systematically, addressing one entry at a time and adapting to user feedback.

### Continuous Feedback:
- Be responsive to user guidance, ready to modify probability estimates according to their suggestions.

### Completion:
- After completing the CPD table, provide a summary and thank the user for their collaboration.

Note: Replace '?' with the suggested probabilities or leave it as a placeholder for the user to fill in. Use this as a template for the visual representation and not for the structure inside..
"""
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
                st.session_state['pattern'] = self.edges_pattern
                st.session_state['nodes'] = match.group(1)
            elif st.session_state.category == self.EDGES:
                st.session_state.category = self.PROBABILITY
                st.session_state['pattern'] = self.probability_pattern
                start_index = match.group(1).find('[')
                end_index = match.group(1).find(']')
                st.session_state['edges'] = match.group(1)[start_index:end_index+1]
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
    # Pattern to extract the dictionary
    node_pattern = r'\{(?:\s*"[^"]+"\s*:\s*\[[^\]]*\]\s*,?)*\s*\}'
    node_match = re.search(node_pattern, response)
    start_index = response.find('[')
    end_index = response.find(']')
    
    if node_match:
        dictionary_str = node_match.group()
        dictionary = json.loads(dictionary_str)
        
        # Get the original text before the dictionary
        original_text = response[:node_match.start()]
        
        # Convert the dictionary to a Markdown table
        table = "| Category | Nodes |\n| --- | --- |\n"
        for key, values in dictionary.items():
            values_str = ", ".join(values)
            table += f"| {key} | {values_str} |\n"
        
        # Get the text after the dictionary
        text_after_dict = response[node_match.end():]
        
        # Combine the original text, Markdown table, and text after the dictionary
        combined_segment = f"{original_text}\n{table}\n{text_after_dict}"
        
        # Return the combined segment
        return [combined_segment]

    elif start_index != -1 and end_index != -1:
        segments = []
        last_end = 0
        edges_str = response[start_index:end_index+1]
        edges = ast.literal_eval(edges_str)
            
        
        index = response.find(edges_str)
        
        if index != -1:
            segments.append(response[0:index])
            last_end = index + len(edges_str)
        
        # Create the graph and append it as the last segment
        graph = create_graph(edges)
        segments.append(graph)
        
        # Append text after the edges string
        segments.append(response[last_end:])
        
        return segments

    else:
        return [response]


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
            query = 'Following are edges collected from dynamically collected data. Incorporate them in your response.{}.\nThis query does not come from the user.'.format(st.session_state['tentative_edges'])
            response = risk_bot.edges_handler.get_response(query)
            st.session_state['responses'].append(response)
    if st.session_state['edges']:
        extract_and_format_edges(st.session_state['edges'])
        if not st.session_state['tentative_cpds']:
            st.session_state['tentative_cpds'] = get_cpds(st.session_state['edges'])
            query = 'Following are CPDs collected from dynamically collected data. Incorporate them in your response.{}\nThis query does not come from the user.'.format(st.session_state['tentative_cpds'])
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



