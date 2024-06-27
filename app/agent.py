# Import necessary libraries and modules
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

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Set the model version based on the current date
current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# Load raw documents from a text file
raw_documents = TextLoader("../data/data.txt").load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
docs = text_splitter.split_documents(documents)

# Further split documents with additional configurations
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

# Initialize the language model with specified parameters
llm = ChatOpenAI(temperature=0.0, model=llm_model, max_tokens=2048)

# Create an in-memory search database from documents and embeddings
db = DocArrayInMemorySearch.from_documents(all_splits, embeddings)
retriever = db.as_retriever()

# Create a RetrievalQA chain for question answering
qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)

# Define a custom tool to find seminar quotes
@tool
def find_seminar_quote(text: str) -> str:
    """
    Receives a question from the user about a seminar and returns a description of it from the transcript.
    """
    query = f"""Please gather more information about '''{text}''' and rephrase the output, and always send me a string"""
    response = qa_stuff.run(query)
    print("find_seminar_quote", response, flush=True)
    return response

# Function to create an agent
def create_agent():
    prefix = """Your name is Christian. You are a specialist agent in teaching Technology. You should have a simple and friendly conversation with a human, answering their questions as best as possible. You have access to the following tools: """
    suffix = """
    history: {chat_history}
    Question: {input}
    {agent_scratchpad}"""
    tools = [find_seminar_quote]

    # Create a prompt for the ZeroShotAgent
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    # Initialize memory for conversation history
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

    # Create a language model chain with the prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Initialize the agent with the LLM chain and tools
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, max_iterations=3)

    # Create an agent executor with error handling
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors="Check if your output is helpful for the question asked."
    )
    print('Agent created successfully')
    return agent_executor

# Example query (commented out for now)
# text = "hologram data size how to transport"
# query = f"""Please gather more information about '''{text}''' and summarize the output"""
# print(qa_stuff.run(query))
