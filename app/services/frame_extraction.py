import cv2
import os

def extract_frames_with_metadata(video_path, output_dir='./extracted_frames', frame_rate=0.5):
    """
    Converts video into frames according given frame rate and saves extracted frames in the folder default or given folder
    """
    os.makedirs(output_dir, exist_ok=True)
    vid_cap = cv2.VideoCapture(video_path)
    
    fps = round(vid_cap.get(cv2.CAP_PROP_FPS))
    count = 0
    frame_data = [] # Store dictionaries with path and time
    
    while True:
        success, image = vid_cap.read()
        if not success:
            break
            
        # Extract 1 frame per 'frame_rate' seconds
        if count % (fps // frame_rate) == 0:
            # 1. Get the exact timestamp in milliseconds from OpenCV
            milli_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
            seconds = round(milli_time / 1000, 2) # Convert to seconds
            
            frame_name = f"frame_{count}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, image)
            
            # 2. Store the path and the time together
            frame_data.append({
                "path": frame_path,
                "timestamp": f"{seconds}s"
            })
            
        count += 1
        
    vid_cap.release()
    return frame_data

# [{'path': './extracted_frames/frame_0.jpg', 'timestamp': '0.0s'}, ...]
# saved_frames = extract_frames_with_metadata("C:/Users/HP-PC/Downloads/Tensecondscounter.mp4")
# print(saved_frames)


