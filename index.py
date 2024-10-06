#######################################################
# Proprieatary Software
#########################################################3
from time import asctime
import asyncio
import cv2
import os
import threading
import time
import requests
import numpy as np
import queue
from fastapi import FastAPI
from fastapi.responses import StreamingResponse,JSONResponse
from deepface import DeepFace
from imageai.Detection import ObjectDetection
from insightface.app import FaceAnalysis

# Folder for saved recordings
RECORDINGS_FOLDER = "recordings"
if not os.path.exists(RECORDINGS_FOLDER):
    os.makedirs(RECORDINGS_FOLDER)

# Global variables
current_frame = None  # Store the current frame for live streaming
is_recording = False  # Flag to control recording
DATA = []  # To store AI detection results
detection_queue = queue.Queue()  # Queue for background AI detection
detector = ObjectDetection()  # Object detection instance

# API key for plate recognition service
AP_KEY = "2a1289ef47a10a6d9a3d53ae608f4c545c13334b"

# Initialize FaceAnalysis
face_analysis_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))

def setup_detector():
    detector.setModelTypeAsRetinaNet()
    model_path = os.path.join(os.getcwd(), "retinanet_resnet50_fpn_coco-eeacb38b.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    detector.setModelPath(model_path)
    try:
        detector.loadModel()
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
def generate_frames():
    global current_frame, is_recording, video_writer
    cam = cv2.VideoCapture(0)

    # Ensure video writer is initialized
    if is_recording and video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDINGS_FOLDER, f"recording_{timestamp}.avi")
        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        current_frame = frame

        # If recording, save the frames
        if is_recording and video_writer:
            video_writer.write(frame)

        # Save frame for AI processing in the background
        _PATH = "index.png"
        cv2.imwrite(_PATH, frame)
        detection_queue.put(_PATH)  # Add to the queue for background processing

        # Sleep to reduce CPU usage
        time.sleep(0.1)

    cam.release()
    if video_writer:
        video_writer.release()  # Ensure the writer is released when done

def start_recording():
    global is_recording, video_writer
    if not is_recording:
        is_recording = True
        
        # Create a filename based on the current timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDINGS_FOLDER, f"recording_{timestamp}.avi")
        
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

        print(f"Recording started: {filename}")
    else:
        print("Recording is already in progress.")

def stop_recording():
    global is_recording, video_writer
    if is_recording:
        is_recording = False
        video_writer.release()
        print("Recording stopped")

def convert_to_serializable(data):
    """Recursively convert non-serializable types to serializable ones."""
    if isinstance(data, np.float32):
        return float(data)  # Convert numpy float32 to Python float
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    return data
def ai_detection_task(image_path):
    try:
        checked_objs = detector.detectObjectsFromImage(input_image=image_path)
        for obj in checked_objs:
            if obj['name'] == 'person':
                print('Detected Person! running deepface')
                data_level_1 = DeepFace.analyze(
                    img_path=image_path,
                    actions=['age', 'gender', 'race'],
                    enforce_detection=False
                )
                print('Deepface complete! Running face analysis...')
                data_level_2 = generate_information(image_path)

                # Convert data to be serializable
                data_level_1 = convert_to_serializable(data_level_1)
                data_level_2 = convert_to_serializable(data_level_2)

                DATA.append({
                    "name": "person",
                    "date": asctime(),
                    "data": {
                        "deepface": data_level_1,
                        "face_analysis": data_level_2
                    }
                })
            elif obj['name'] == 'car':
                print('Detected Car... running plate recognition')
                headers = {"Authorization": f"Token {AP_KEY}"}
                with open(image_path, 'rb') as fp:
                    response = requests.post(
                        'https://api.platerecognizer.com/v1/plate-reader/',
                        headers=headers,
                        files={"upload": fp}
                    )
                DATA.append({
                    "name": "car",
                    "date": asctime(),
                    "data": response.json()
                })
    except Exception as e:
        print(f"Error during AI detection: {str(e)}")
def generate_information(ip):
    img = cv2.imread(ip)
    if img is None:
        return {"error": "image could not be loaded"}
    
    faces = face_analysis_app.get(img)
    dt_out = []
    for idx, face in enumerate(faces):
        print(f"Face {idx + 1}:")
        data = {}
        bbox = list(face.bbox)  # Ensure accessing via attribute notation
        print(f"  Bounding Box: {bbox}")
        data['bbox'] = bbox
        landmarks = list(face.kps)  # key points
        print(f"  Landmarks: {landmarks}")
        data['landmarks'] = landmarks
        
        gender = getattr(face, 'gender', 'N/A')  # Use getattr to avoid key errors
        age = getattr(face, 'age', 'N/A')
        print(f"  Gender: {'Male' if gender != 1 else 'Female'}")
        print(f"  Age: {age}")
        data['age'] = age
        data['gender'] = 'Male' if gender != 1 else 'Female'
        
        dt_out.append(data)
    return dt_out

def background_detection():
    while True:
        image_path = detection_queue.get()
        if image_path:
            ai_detection_task(image_path)
        detection_queue.task_done()

app = FastAPI()

@app.get("/recordings")
async def list_recordings():
    files = os.listdir(RECORDINGS_FOLDER)
    return JSONResponse({"recordings": files})

@app.get("/recordings/{filename}")
async def get_recording(filename: str):
    file_path = os.path.join(RECORDINGS_FOLDER, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="video/avi")

@app.get("/video_feed")
async def video_feed():
    async def frame_generator():
        global current_frame
        while True:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame_data = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            await asyncio.sleep(0.1)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
@app.get('/data')
async def data():
    return JSONResponse(DATA)

def start_fastapi():
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
if __name__ == "__main__":
    if not setup_detector():
        print('Could not setup detector')
        exit(1)
    start_recording()
    # Start background detection thread
    background_thread = threading.Thread(target=background_detection, daemon=True)
    background_thread.start()

    # Start frame generation thread
    frame_thread = threading.Thread(target=generate_frames, daemon=True)
    frame_thread.start()

    # Start FastAPI server
    fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
    fastapi_thread.start()

    fastapi_thread.join()
else:
    print('Denied Acess to start program')