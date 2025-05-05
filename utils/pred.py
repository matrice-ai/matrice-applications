import requests
import cv2
import os
import shutil
import time

def process_video(video_path, auth_key, output_path="output.mp4", temp_dir="temp_frames"):
    """
    Process a video frame by frame, send each frame to the API, 
    draw bounding boxes based on predictions, and save as a new video.
    
    Args:
        video_path (str): Path to the input video file
        auth_key (str): Authentication key for the API
        output_path (str): Path for the output video
        temp_dir (str): Directory to store temporary frames
    """
    # Create temporary directory for frames
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_folder = os.path.join(current_dir, temp_dir)
    
    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    else:
        # Clear existing files
        for file in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, file))
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(current_dir, output_path), fourcc, fps, (width, height))
    
    frame_index = 0
    
    # Process each frame
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        print(f"Processing frame {frame_index}/{total_frames}")
        
        # Save the frame to the temp directory
        frame_path = os.path.join(temp_folder, f"frame_{frame_index:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Process the frame with the API
        try:
            # Open the file for the API request
            files = {'input': open(frame_path, 'rb')}
            
            # Prepare the data
            data = {
                'authKey': auth_key,
            }
            
            headers = {}
            
            # API URL
            url = "https://prod.backend.app.matrice.ai/v1/model_prediction/deployment/6810f66744e907b404a79b4d/predict"
            
            # Make the API request
            response = requests.post(url, headers=headers, data=data, files=files)
            
            # Close the file handle
            files['input'].close()
            
            # Parse the API response
            if response.status_code == 200:
                result = response.json()
                
                # Check if we have valid predictions
                if result.get('success') and 'data' in result:
                    predictions = result['data']
                    
                    # Draw bounding boxes for each prediction
                    for pred in predictions:
                        category = pred.get('category')
                        confidence = pred.get('confidence')
                        bbox = pred.get('bounding_box')
                        
                        if bbox:
                            xmin = bbox.get('xmin')
                            ymin = bbox.get('ymin')
                            xmax = bbox.get('xmax')
                            ymax = bbox.get('ymax')
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            
                            # Add text with category and confidence
                            label = f"{category}: {confidence:.2f}"
                            cv2.putText(frame, label, (xmin, ymax + 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Write the processed frame to the output video
            out.write(frame)
            
        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")
        
        # Increment frame index
        frame_index += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_folder)
    
    print(f"Processing complete. Output saved to {output_path}")

def process_single_frame(video_path, frame_index=0, auth_key="AUTH_KEY"):
    """
    Process a single frame from a video and send it to the API for prediction.
    
    Args:
        video_path (str): Path to the video file
        frame_index (int): Index of the frame to process (0 = first frame)
        auth_key (str): Authentication key for the API
    
    Returns:
        dict: The API response
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_index}")
        return None
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the frame to a file in the same directory
    frame_path = os.path.join(current_dir, f"frame_{frame_index}.jpg")
    cv2.imwrite(frame_path, frame)
    
    print(f"Frame saved at: {frame_path}")
    
    # Check if the file exists and can be opened
    if not os.path.exists(frame_path):
        print(f"Error: File does not exist at {frame_path}")
        return None
        
    try:
        with open(frame_path, 'rb') as test_file:
            print(f"Successfully opened file at {frame_path}")
    except Exception as e:
        print(f"Error opening file: {e}")
        return None
    
    # API URL
    url = "https://prod.backend.app.matrice.ai/v1/model_prediction/deployment/6810f66744e907b404a79b4d/predict"
    
    # Open the file for the API request
    files = {'input': open(frame_path, 'rb')}
    
    # Prepare the data
    data = {
        'authKey': auth_key,
    }
    
    headers = {}
    
    # Make the API request
    response = requests.post(url, headers=headers, data=data, files=files)
    
    # Close the file handle
    files['input'].close()
    
    # Return the response
    return response.json() if response.status_code == 200 else response.text

if __name__ == "__main__":
    # Replace these values with your actual values
    VIDEO_PATH = r"C:\Users\pathi\OneDrive\Desktop\matriceai\matrice-applications\utils\gas_flare_video.mov"
    AUTH_KEY = "6811ede444e907b404a7a182"
    
    # Choose one of the two modes:
    
    # 1. Process a single frame (for testing)
    # result = process_single_frame(VIDEO_PATH, frame_index=0, auth_key=AUTH_KEY)
    # print("API Response:")
    # print(result)
    
    # 2. Process the entire video
    process_video(VIDEO_PATH, AUTH_KEY, "output.mp4", "temp_frames")