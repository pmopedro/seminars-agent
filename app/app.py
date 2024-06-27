"""
Demonstrates how to use the `ChatInterface` and `PanelCallbackHandler` to create a chatbot to talk to your Pandas DataFrame. This is heavily inspired by the
[LangChain `chat_pandas_df` Reference Example](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_pandas_df.py).
"""
from __future__ import annotations
from agent import llm_model
import datetime
import os
from pathlib import Path
from textwrap import dedent
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
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from agent import create_agent

# Initialize panel extension
pn.extension("perspective")

class AgentConfig(param.Parameterized):
    """Configuration class for the Pandas Agent"""
    user = param.String("Professor Agent")
    avatar = param.String("🧑‍🏫")
    show_chain_of_thought = param.Boolean(default=False)

    def _get_agent_message(self, message: str) -> pn.chat.ChatMessage:
        """Creates a chat message with agent's user and avatar settings"""
        return pn.chat.ChatMessage(message, user=self.user, avatar=self.avatar)

class AppState(param.Parameterized):
    """Application state for managing LLM and agent instances"""
    llm = param.Parameter()
    agent = param.Parameter()
    config: AgentConfig = param.ClassSelector(class_=AgentConfig)

    def __init__(self, config: AgentConfig | None = None):
        """Initialize application state with default or provided configuration"""
        if not config:
            config = AgentConfig()
        super().__init__(config=config)
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        print("OPENAI_API_KEY:", self.OPENAI_API_KEY)
        self._reset_llm()
        self._reset_agent()

    def _reset_llm(self):
        """Reset the language model with OpenAI API key and configurations"""
        if self.OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    temperature=0,
                    model="gpt-3.5-turbo-1106",
                    api_key=self.OPENAI_API_KEY,
                    streaming=True,
                )
                print("LLM successfully instantiated")
            except Exception as e:
                print(f"Error instantiating LLM: {e}")
                self.llm = None
        else:
            print("OPENAI_API_KEY is not set")
            self.llm = None

    @param.depends("llm", on_init=True, watch=True)
    def _reset_agent(self):
        """Reset the agent based on LLM availability"""
        with param.edit_constant(self):
            if not self.error_message:
                self.agent = create_agent()
            else:
                self.agent = None

    @property
    def error_message(self):
        """Returns an error message if LLM is not available"""
        if self.llm is None:
            return dedent(
                """\
                LLM not detected, please contact support."""
            )
        return ""

    @property
    def welcome_message(self):
        """Returns a welcome message including any error messages"""
        return dedent(
            f"""
            Hey, my name is Chris. I'm a professor agent and I'm here to solve your questions about seminars.
            {self.error_message}"""
        ).strip()

    async def callback(self, contents, user, instance):
        """Handles chat interface callbacks with user inputs and generates responses"""
        print("Content: ", contents)
        print("User: ", user)
        print("Instance: ", instance)
        if not isinstance(contents, str):
            self.data = contents
            instance.active = 1
            message = self.config._get_agent_message(
                dedent(
                    """Hey, my name is Chris. I'm a professor agent and I'm here to solve your questions about seminars."""
                )
            )
            return message

        if self.error_message:
            message = self.config._get_agent_message(self.error_message)
            return message

        if self.config.show_chain_of_thought:
            langchain_callbacks = [
                pn.chat.langchain.PanelCallbackHandler(instance=instance)
            ]
        else:
            langchain_callbacks = []
        print({"inputs": contents})

        response = await self.agent.ainvoke(
            {"input": contents}, callbacks=langchain_callbacks
        )
        message = self.config._get_agent_message(response['output'])
        return message

# Initialize application state
state = AppState()

# Set up chat interface
chat_interface = pn.chat.ChatInterface(
    widgets=[
        pn.widgets.TextInput(name="Message", placeholder="Send a message"),
    ],
    renderers=pn.pane.Perspective,
    callback=state.callback,
    callback_exception="verbose",
    show_rerun=False,
    show_undo=False,
    show_clear=False,
    min_height=300,
)
chat_interface.send(
    state.welcome_message,
    user=state.config.user,
    avatar=state.config.avatar,
    respond=False,
)

# Set up layout for the chat interface
layout = pn.template.MaterialTemplate(
    title="Intelligent Auto Parts Agent",
    main=[chat_interface],
    sidebar=[
        "#### Agent Settings",
        state.config.param.show_chain_of_thought,
    ],
)

# Make the layout servable
layout.servable()
