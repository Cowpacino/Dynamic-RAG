from langchain_core.tools import tool
from langsmith import traceable
import base64
import cv2
import os

from app.core.vector_store import video_store_manager


@tool
@traceable(name="video_content_from_vector_store")
def retrieve_video_content_from_vector_store(query: str):
    """
    If search content is related to video then Search for content internal Knowledge of Videos.
    Use this tool when the user asks questions about uploaded Video or previously stored video.
    """
    try:
        # Perform similarity search using the manager
        retrieved_docs = video_store_manager.similarity_search(query, k=1)

        if not retrieved_docs:
            message_content = f"No relevant information found in the local knowledge base for: '{query}'."
            return (message_content, [])
        
        image_data_list = final_result(retrieved_docs)
        context_info = " | ".join([f"Frame {i}: {d['timestamp']}" for i, d in enumerate(image_data_list)])

        
        
        message_content = f"""
        You are a vision-capable AI.
        Frame information:
        {context_info}
        Question:
        {query}
        Answer based ONLY on the video frames.
        """
                
        # for item in image_data_list:
        #     base64_image = encode_image(item['path'])
            # content.append({
            #     "type": "image_url",
            #     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            # })
        message_content = message_content

        # Return tuple: (message_content for LLM, raw documents as artifact)
        return (message_content, retrieved_docs)
    except Exception as e:
        error_message = f"Error retrieving context from vector store: {str(e)}"
        return (error_message, [])
    


     
def encode_image(image_path,target_width=150):
    """ Convert images into base64encode to process"""
    img = cv2.imread(image_path)
    if img is None:
        return ""
    
    # Calculate new dimensions
    height, width = img.shape[:2]
    new_height = int(target_width * (height / width))
    resized = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Encode directly from the array to base64
    _, buffer = cv2.imencode('.jpg', resized)
    return base64.b64encode(buffer).decode('utf-8')
    
    
def final_result(responses):
    
    image_data_list = []
    for path, meta in zip(responses['uris'][0], responses['metadatas'][0]):
        image_data_list.append({'path': path, 'timestamp': meta['timestamp']})

    return image_data_list 

