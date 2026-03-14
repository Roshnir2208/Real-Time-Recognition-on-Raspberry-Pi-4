import cv2
import os
import time
from picamera2 import Picamera2

# Ask for user name
person_name = input("Enter the person's name: ").strip()
SAVE_DIR = "known_faces"
person_dir = os.path.join(SAVE_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)
print(f"Saving images to: {person_dir}")

# Initialize Picamera2
picam2 = Picamera2()

# Use preview configuration which applies full color processing
camera_config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)  # Allow AWB and exposure to stabilize

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
TOTAL_IMAGES = 40

print("\nStarting capture... Press Ctrl+C to quit early.\n")

try:
    while count < TOTAL_IMAGES:
        # Capture frame (this is already color corrected)
        image = picam2.capture_array()
        
        # Convert XRGB8888 → BGR for OpenCV
        #image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capture Faces", image)

        # Save first detected face
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = image[y:y+h, x:x+w]
            filename = os.path.join(person_dir, f"img_{count:04d}.jpg")
            cv2.imwrite(filename, face_crop)
            print(f"Saved {filename}")
            count += 1
            time.sleep(0.15)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

cv2.destroyAllWindows()
picam2.stop()
print(f"Finished! Saved {count} images for {person_name}.")
