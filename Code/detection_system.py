import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
import os
import sys
import json

# --------------------------
# CONFIG & MODEL LOADING
# --------------------------

# Face Recognition Settings
CV_SCALER = 4       # Factor to downsize the frame for face detection (1/4 size)
RECOGNITION_THRESHOLD = 0.60 # Max distance for a match (0.6 is common for Dlib)

# YOLOv4-tiny Settings
YOLO_CFG = "yolov4-tiny.cfg"
YOLO_WEIGHTS = "yolov4-tiny.weights"
LABELS_PATH = "coco.names"
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_NMS_THRESHOLD = 0.4
IGNORE_OBJECTS = {"person"} 

# Frame and Loop Settings
FRAME_WIDTH, FRAME_HEIGHT = 480, 270
OBJECT_DETECTION_INTERVAL = 5 # Run YOLO every 5 frames
TARGET_FPS = 20

# Video Output Settings
OUTPUT_FPS = 10 
FOURCC = cv2.VideoWriter_fourcc(*"MJPG") 

# Load pre-trained face encodings
print("[INFO] Loading face encodings...")
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
except FileNotFoundError:
    print("Error: encodings.pickle not found. Please run your encoding script first.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading encodings: {e}")
    sys.exit(1)

# Load YOLO model
net = None
classes = []
output_layers = []
print("[INFO] Loading YOLO model...")
try:
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(LABELS_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_out_layers, np.ndarray):
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    
except Exception as e:
    print(f"Warning: YOLO files not loaded. Object detection disabled. Error: {e}")
    net = None

# --------------------------
# SCENARIO HANDLING
# --------------------------
def apply_scenario(frame, scenario):
    """Applies basic image adjustments for scenario testing."""
    
    if scenario == "bright_room":
        return cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    elif scenario == "outdoor_evening":
        return cv2.convertScaleAbs(frame, alpha=0.8, beta=-20)
    elif scenario == "strong_backlight":
        return cv2.convertScaleAbs(frame, alpha=0.7, beta=-30)
    
    return frame

# --------------------------
# YOLO OBJECT DETECTION FUNCTION
# --------------------------
def run_object_detection(frame, net, output_layers):
    """
    Performs YOLOv4-tiny detection on the full-size frame.

    """
    if net is None:
        return []

    H, W = frame.shape[:2]
    
    
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > YOLO_CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_THRESHOLD)
    
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            class_name = classes[class_ids[i]]
            
            if class_name in IGNORE_OBJECTS: 
                continue

            x, y, w, h = boxes[i]
            x2, y2 = x + w, y + h

            results.append({
                "label": class_name,
                "confidence": confidences[i],
                "bbox": [x, y, x2, y2]
            })
            
    return results

# --------------------------
# MAIN DETECTION LOOP
# --------------------------
def run_detection(scenario="default"):
    # Initialize the camera
    print(f"\nRunning scenario: {scenario}\n")
    picam2 = Picamera2()
    loop_start_time = time.time() 

    cv2.namedWindow('Detection System', cv2.WINDOW_NORMAL)
    
    
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    
    # Set controls before starting
    picam2.set_controls({"Brightness": 0.0, "Contrast": 1.0 })
    picam2.start()
    
    # Allow camera to settle
    time.sleep(2.0)

    # ----------------------------------------------------
    #  Video Writer Initialization
    # ----------------------------------------------------
    OUTPUT_FILENAME = f"detection_output_{scenario}_{time.strftime('%Y%m%d_%H%M%S')}.avi"
    out = None
    try:
        
        out = cv2.VideoWriter(
            OUTPUT_FILENAME,
            FOURCC,
            OUTPUT_FPS,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )
        print(f"[INFO] Initialized video writer: {OUTPUT_FILENAME} at {OUTPUT_FPS} FPS.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize VideoWriter: {e}. Video saving disabled.")

    # Initialize variables
    face_locations = []
    face_names = []
    yolo_results = []
    log = []
    frame_count = 0
    start_time = time.time()
    fps = 0

    cv2.namedWindow('Detection System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection System', FRAME_WIDTH, FRAME_HEIGHT)
    
    while True:
        frame = picam2.capture_array() 
        current_frame_time = time.time() - start_time
        
        # Apply scenario filter
        frame = apply_scenario(frame, scenario)
        
        # ----------------------------------------------------
        # 1. FACE RECOGNITION (Runs every frame on scaled image)
        # ----------------------------------------------------
        
        # Resize frame. NO COLOR CONVERSION NEEDED: Frame is already RGB!
        resized_frame = cv2.resize(frame, (0, 0), fx=(1/CV_SCALER), fy=(1/CV_SCALER))
        
        # Detect faces
        face_locations = face_recognition.face_locations(resized_frame, model="hog")
        face_encodings = face_recognition.face_encodings(resized_frame, face_locations, model='large')
        
        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] < RECOGNITION_THRESHOLD: 
                name = known_face_names[best_match_index]
                confidence = (1 - face_distances[best_match_index]) * 100
                face_names.append(f"{name} ({confidence:.1f}%)")
            else:
                face_names.append("Unknown")

        # ----------------------------------------------------
        # 2. OBJECT DETECTION 
        # ----------------------------------------------------
        if net and frame_count % OBJECT_DETECTION_INTERVAL == 0:
            
            yolo_results = run_object_detection(frame, net, output_layers)
            
        # ----------------------------------------------------
        # 3. DRAW RESULTS (Draws on RGB frame)
        # ----------------------------------------------------
        display_frame = frame.copy()

        # Draw Face Recognition Results (Red box on RGB frame)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= CV_SCALER
            right *= CV_SCALER
            bottom *= CV_SCALER
            left *= CV_SCALER
            # Extract name and confidence from the formatted string
            parts = name.split(' (')
            label = parts[0]
            confidence_str = parts[1].strip('%)') if len(parts) > 1 else '100'
            confidence = float(confidence_str) / 100.0 if confidence_str != 'Unknown' else 0.0

            
            log.append({
                "time": current_frame_time,
                "type": "face",
                "label": label,
                "confidence": confidence,
                "bbox": [int(left), int(top), int(right), int(bottom)]
            })
            # The draw functions work fine on RGB data, but expect BGR color order (B, G, R) for the tuple. 
            # (255, 0, 0) in RGB space is RED.
            cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.rectangle(display_frame, (left -3, top - 35), (right+5, top), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(display_frame, name, (left + 6, top - 6), font, 0.8, (0, 0, 255), 1)

        # Draw YOLO Object Detection Results (Green box on RGB frame)
        for obj in yolo_results:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            confidence = obj["confidence"]
            
            log.append({
                "time": current_frame_time,
                "type": "object",
                "label": label,
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            
            # (0, 255, 0) in RGB space is GREEN.
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{label} ({confidence:.1%})", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ----------------------------------------------------
        # 4. FPS, VIDEO WRITE, AND DISPLAY
        # ----------------------------------------------------
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        

        if out:
            out.write(display_frame) 
            
        cv2.imshow('Detection System', display_frame)
        
        if cv2.waitKey(1) == ord("q"):
            break
            
        if cv2.waitKey(1) == ord("q"):
            break

        # --- ENFORCE TARGET FRAME RATE ---
        # Calculate the time spent in this frame iteration
        frame_processing_time = time.time() - loop_start_time 
        
        # Calculate the required delay to hit the target FPS
        delay_time = (1.0 / TARGET_FPS) - frame_processing_time
        
        # Only sleep if the processing time was less than the target frame duration
        if delay_time > 0:
            time.sleep(delay_time) 
        
        loop_start_time = time.time() # Reset the timer for the next loop    

    # Cleanup
    if out:
        out.release()
        print(f"[INFO] Video saved to {OUTPUT_FILENAME}")
        
    cv2.destroyAllWindows()
    picam2.stop()
    LOG_FILENAME = f"log_{scenario}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(LOG_FILENAME, "w") as f:
            json.dump(log, f, indent=2)
        print(f"[INFO] Log saved to {LOG_FILENAME}")
    except Exception as e:
        print(f"[ERROR] Failed to save log file: {e}")

    print(f"Scenario '{scenario}' completed!")

# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        run_detection("default")
    else:
        scenario = sys.argv[1]
        run_detection(scenario)
