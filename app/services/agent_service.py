from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

from app.core.config import settings
from app.tools.browse_tool import browse_webpage
from app.tools.retrieval_tool import retrieve_from_vector_store
from app.tools.web_search_tool import search_the_internet


class AgentService:
    """
    Service responsible for initializing and interacting with the ReAct agent.
    The agent can choose between local PDF/Web knowledge or searching the live internet.
    """

    def __init__(self):
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            api_key=SecretStr(settings.OPENAI_API_KEY)
            if settings.OPENAI_API_KEY
            else None,
            temperature=0,
        )

        # Define the tools available to the agent
        self.tools = [retrieve_from_vector_store, search_the_internet, browse_webpage]

        # System prompt to guide the agent's behavior
        self.system_prompt = (
            "You are a highly capable modular RAG assistant. "
            "You have access to three main tools:\n"
            "1. retrieve_from_vector_store: Use this to find information in documents (PDFs or Webpages) that have been uploaded or indexed locally.\n"
            "2. search_the_internet: Use this to find real-time information or when local knowledge is insufficient.\n"
            "3. browse_webpage: Use this to visit a specific URL, parse its content, and extract information directly.\n\n"
            "Always try to provide accurate, concise, and helpful answers. "
            "If you use a tool, cite the source information provided in the tool output."
        )

        # Create the LangGraph ReAct agent
        self.agent_executor = create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
        )

    async def chat(self, message: str, thread_id: str = "default"):
        """
        Sends a message to the agent and returns the response.
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        input_state = {"messages": [HumanMessage(content=message)]}

        try:
            # Execute the agent and get the state
            result = await self.agent_executor.ainvoke(input_state, config=config)

            # The last message in the state will be the AI's response
            return result["messages"][-1].content
        except Exception as e:
            return f"Agent encountered an error: {str(e)}"


# Global instance
agent_service = AgentService()
