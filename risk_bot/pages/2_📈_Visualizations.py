import streamlit as st
import re
import ast
from html import escape
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
import streamlit.components.v1 as components

with open("styles.css") as f:                                                
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html= True) 

openai_api_key = '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'



def check_regex_pattern(text):
    pattern = r'(?i)Done with all the probability values.'
    return re.search(pattern, text) 

def process_text_with_llm(text):
    llm = ChatOpenAI(temperature=0, openai_api_key= openai_api_key, model='gpt-4-1106-preview')

    visualisation_template = """
    You are a visualisation expert. Here is the text: {text} , go thorugh this and follow the steps:
    step 1. This entire text is a conversation about finalising some nodes, edges and conditional probability distribution for a bayesain network. 
    step 2. You have to find all conditional probability distributions from the text (this could involve parsing the user's input for all these values). 
    step 3: i want to display the CPD for each variable as st.bar_chart() on a streamlit page."
    step 4: create one plot for each variable. if the variable has parents, then create a stacked bar chart. if the variable has no parents, then create a simple bar chart.
    step 5. Create an end to end end python script for all the CPD visualisations.
    step 6. Return this python script only
    REMEMBER ONLY RETURN A PYTHON SCRIPT WHICH HAS ALL THE VISUALISATION CODE IN PYTHON FOR A STREAMIT PAGE with the data i provided in the text. 
    REMEMBER: DO not call st.set_page_config() in the python script.
    """

    prompt_template = PromptTemplate(input_variables=["text"], template=visualisation_template)
    prompt_template.format(text=text)

    chain = LLMChain(llm=llm,
                    prompt=prompt_template)
    output = chain.invoke(text)

    print(output['text'])

    pattern = re.compile(r"```python(.*?)```", re.DOTALL)
    matches = pattern.findall(output['text'])



    return matches


if st.session_state['nodes'] is not None:
    nodes = ast.literal_eval(st.session_state['nodes'])
    
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
    st.write("""<h1>Nodes</h1>""", unsafe_allow_html=True)
    st.write(table, unsafe_allow_html=True)

if st.session_state['edges'] is not None:
    HtmlFile = open(f'graph.html', 'r', encoding='utf-8')
    st.write("""<h1>Edges</h1>""", unsafe_allow_html=True)
    components.html(HtmlFile.read(), height=435)

if check_regex_pattern(st.session_state['buffer_memory'].buffer_as_str) is not None:
    matches = process_text_with_llm(st.session_state['buffer_memory'].buffer_as_str)
    st.write("""<h1>Probabilities</h1>""", unsafe_allow_html=True)
    for match in matches:
        exec(match)


