from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
import logging
from app.core.config import settings
import base64
import cv2
from app.tools.browse_tool import browse_webpage
from app.tools.retrieval_tool import retrieve_from_vector_store
from app.tools.web_search_tool import search_the_internet
from app.tools.video_retrieval_tool import retrieve_video_content_from_vector_store


class AgentService:
    """
    Service responsible for initializing and interacting with the ReAct agent.
    The agent can choose between local PDF/Web knowledge or searching the live internet.
    """

    def __init__(self):
        # Initialize the LLM
        # self.text_model = ChatOpenAI(
        #     model=settings.MODEL_NAME,
        #     api_key=SecretStr(settings.OPENAI_API_KEY)
        #     if settings.OPENAI_API_KEY
        #     else None,
        #     temperature=0,
        # )

        self.text_model = ChatOllama(
            model="qwen2.5:3b",
            temperature=0
        )
        self.vlm_model = ChatOllama(
            model="qwen3-vl:2b",
            temperature=0
        )

        # Define the tools available to the agent
        self.tools = [retrieve_from_vector_store, search_the_internet, browse_webpage,retrieve_video_content_from_vector_store]

        # System prompt to guide the agent's behavior
        self.system_prompt = (
            "You are a highly capable modular RAG assistant. "
            "You have access to three main tools:\n"
            "1. retrieve_from_vector_store: Use this to find information in documents (PDFs or Webpages) that have been uploaded or indexed locally.\n"
            "2. search_the_internet: Use this to find real-time information or when local knowledge is insufficient.\n"
            "3. browse_webpage: Use this to visit a specific URL, parse its content, and extract information directly.\n\n"
            "4. retrieve_video_content_from_vector_store: Use this when query related to an video that have been uploaded or indexed locally.\n\n"
            "Always try to provide accurate, concise, and helpful answers. "
            "If you use a tool, cite the source information provided in the tool output."
        )

        # Create the LangGraph ReAct agent
        self.agent_executor = create_react_agent(
            self.text_model,
            tools=self.tools,
            prompt=self.system_prompt,
        )
    
    # async def chat(self, message: str, thread_id: str = "default"):
    #     """
    #     Sends a message to the agent and returns the response.
    #     """
    #     config: RunnableConfig = {"configurable": {"thread_id": thread_id},"run_name":"Check tools"}
    #     input_state = {"messages": [HumanMessage(content=message)]}

    #     try:
    #         # Execute the agent and get the state
    #         result = await self.agent_executor.ainvoke(input_state, config=config)

    #         # The last message in the state will be the AI's response
    #         return result["messages"][-1].content
    #     except Exception as e:
    #         return f"Agent encountered an error: {str(e)}"

    async def chat(self, message: str, thread_id: str = "default"):

      
        video_keywords = ["video", "frame", "clip", "scene", "timestamp"]

        if any(word in message.lower() for word in video_keywords):
            logging.info("Sending to VLM...")
            tool_output, retrieved_docs = retrieve_video_content_from_vector_store.func(message)

            # Build proper multimodal message for Ollama
            image_data_list = []
            for path, meta in zip(retrieved_docs['uris'][0], retrieved_docs['metadatas'][0]):
                image_data_list.append({'path': path, 'timestamp': meta['timestamp']})

            

            content = [{"type": "text", "text": f"Question: {message}\nAnswer based ONLY on the video frames below."}]
            
            for item in image_data_list:
                img = cv2.imread(item['path'])
                if img is None:
                    continue
                # Resize to reduce tokens
                img = cv2.resize(img, (512, 288))
                _, buffer = cv2.imencode('.jpg', img)
                b64 = base64.b64encode(buffer).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

            response = self.vlm_model.invoke([HumanMessage(content=content)])
            return response.content
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.agent_executor.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )

        return result["messages"][-1].content


# Global instance
agent_service = AgentService()
