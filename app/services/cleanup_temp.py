import os
from app.core.vector_store import video_store_manager
from app.services.frame_extraction import extract_frames_with_metadata
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def cleanup_temporary_file(filepath: str):
    """Utility function to delete the file."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Cleaned up video file: {filepath}")
        except PermissionError:
            print(f"Warning: Could not delete {filepath} right now.")


def process_video_heavy_lifting(filepath: str):
    print(f"Starting long video processing for {filepath}...")
    try:
        extracted_metadata = extract_frames_with_metadata(filepath)
        video_store_manager.add_documents(documents=None, extracted_metadata=extracted_metadata, is_video_processing=True)
    except Exception as e:
        logger.info("Exception: ",e)

    
        



