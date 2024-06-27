from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import panel as pn
import param
import requests
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents import load_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.agents import tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
import os
import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import TextLoader


embeddings = OpenAIEmbeddings()

# account for deprecation of LLM model
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

raw_documents = TextLoader("../data/data.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

docs = text_splitter.split_documents(documents)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)


llm = ChatOpenAI(temperature=0.0, model=llm_model, max_tokens=2048)
db = DocArrayInMemorySearch.from_documents(
    all_splits,
    embeddings
)
retriever = db.as_retriever()
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)


@tool
def find_seminar_quote(text: str) -> str:
    """ Receives a question from the user about a seminar and returns a description of it from the transcript.
    """

    query = f"""Please gather more information about '''{text}''' and rephrase the output, and always send me a string"""
    response = qa_stuff.run(query)
    print("find_seminar_quote", response, flush=True)

    return response


def create_agent():
    prefix = """Your name is Christian. You are a specialist agent in teaching Technology. You should have a simple and friendly conversation with a human, 
    answering their questions as best as possible. You have access to the following tools: """
    # FORMAT_INSTRUCTIONS = """To use a tool, follow this format:
    #     '''
    #     Thought: Do I really need to use this tool?
    #     Action: The action to take, should be one of [{tool_names}]
    #     Action Input: the input for the desired action
    #     Observation: The result of the action
    #     '''
    #     """
    suffix = """
    hystory: {chat_history}
    Question: {input}
    {agent_scratchpad}"""
    tools = [find_seminar_quote]

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        # format_instructions=FORMAT_INSTRUCTIONS,
        input_variables=[
            "input", "chat_history", "agent_scratchpad"],
    )
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")
    # memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(
        llm_chain=llm_chain, tools=tools, verbose=True, max_iterations=3,)
    agent = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors="Check if your output is helpful for the question asked."
    )
    print('criei o agente corretamente')
    return agent


# text = "hologram data size how to transport"
# query = f"""Please gather more information about '''{text}''' and summarize the output"""

# print(qa_stuff.run(query))
