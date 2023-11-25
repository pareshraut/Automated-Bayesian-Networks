import streamlit as st
import re
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

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
    step 1. This entire text is a conversation about finalising some nodes and edges for a bayesain network. 
    step 2. You have to find all finalised nodes and edges and relevant information about them. 
    step 3. Then create conditional probability tables for each of these edges , from the discussion in the text.
    step 4. Focus on the html tables created in the text and use that information carefully. 
    step 5. Nodes should be in a table with class='styled-table', use pyvis for bayesian network graph and use class='styled-table' for st.table
    step 6. I want to display all my visualisation on a strreamlit page.
    step 7. Create an end to end end python script for all the visualisation you think is approprate with streamlit.
    step 8. Return this python script only
    REMEMBER TO USE THE SOMETHING LIKE THIS FOR ALL tables =
                <table class="styled-table">
                <thead>
                    <tr><th>Category</th><th>Nodes</th></tr>
                </thead>
                <tbody>
    REMEMBER TO USE CODE LIKE FOR THE GRAPH YOU MAKE AT THE END: 
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
    REMEMBER ONLY RETURN A PYTHON SCRIPT WHICH HAS ALL THE VISUALISATION CODE IN PYTHON FOR A STREAMIT PAGE with the data i provided in the text. 
    """

    prompt_template = PromptTemplate(input_variables=["text"], template=visualisation_template)
    prompt_template.format(text=text)

    chain = LLMChain(llm=llm,
                    prompt=prompt_template)
    output = chain.invoke(text)

    return output



if check_regex_pattern(st.session_state['buffer_memory']) is not None:
    output = process_text_with_llm(st.session_state['buffer_memory'])

pattern = re.compile(r"```python(.*?)```", re.DOTALL)
matches = pattern.findall(output)
for match in matches:
    exec(match)

