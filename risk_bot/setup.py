from langchain.chains import ConversationChain
from langchain import PromptTemplate, LLMChain

class Nodes:
    def __init__(self, memory, prompt, chat):
        self.conversation = ConversationChain(memory=memory, prompt=prompt, llm=chat, verbose=False)

    def get_response(self, input_text):
        response = self.conversation.predict(input=f"{input_text}")
        return response

class Edges:
    def __init__(self, memory, prompt, chat):
        self.conversation = ConversationChain(memory=memory, prompt=prompt, llm=chat, verbose=False)

    def get_response(self, input_text):
        response = self.conversation.predict(input=f"{input_text}")
        return response

class ProbabilityDistribution:
    def __init__(self, memory, prompt, chat):
        self.conversation = ConversationChain(memory=memory, prompt=prompt, llm=chat, verbose=False)

    def get_response(self, input_text):
        response = self.conversation.predict(input=f"User:\n{input_text}")
        return response

class Decision:
    def __init__(self, memory,prompt, chat):
        self.conversation = ConversationChain(memory = memory, prompt=prompt, llm=chat, verbose=False)

    def get_response(self, input_text):
        response = self.conversation.predict(input=f"User:\n{input_text}")
        return response