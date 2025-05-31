import cv2
from ultralytics import YOLO
import os
import threading
import pygame

# Sound file paths for animals (use raw strings for paths)
sound_files = {
    "elephant": r"C:\Users\Admin\PycharmProjects\PythonProject1\2sounds_animal\2sounds_animal\dentist_drill.wav",
    "tiger": r"C:\Users\Admin\PycharmProjects\PythonProject1\2sounds_animal\2sounds_animal\construction.wav",
    "cheetah": r"C:\Users\Admin\PycharmProjects\PythonProject1\2sounds_animal\2sounds_animal\alarm_beep.wav",
    "lion": r"C:\Users\Admin\PycharmProjects\PythonProject1\2sounds_animal\2sounds_animal\air_raid.wav"
}

# Initialize pygame mixer for sound
pygame.mixer.init()


# Function to play sound in a separate thread
def play_sound(animal_name):
    if animal_name in sound_files:
        sound_file = sound_files[animal_name]
        if os.path.exists(sound_file):
            try:
                # Load the sound file first
                pygame.mixer.music.load(sound_file)

                # Use threading to play sound asynchronously
                threading.Thread(target=pygame.mixer.music.play).start()
            except pygame.error as e:
                print(f"Error playing sound for {animal_name}: {e}")
        else:
            print(f"Error: Sound file {sound_file} not found.")
    else:
        print(f"Error: No sound file available for {animal_name}")


# Path to YOLO model
model_path = r"C:\Users\Admin\PycharmProjects\PythonProject1\best (2).pt"

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

# Load YOLO model
model = YOLO(model_path)

# Open the webcam (replace '0' with your webcam index if necessary)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process the live video feed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Resize the frame to 640x640 (required input size for YOLO)
    frame_resized = cv2.resize(frame, (640, 640))

    # Predict objects in the resized frame using YOLO
    results = model(frame_resized)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        class_names = result.names  # Class names dictionary

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # Scale the bounding box coordinates back to original frame size
            x1 = int(x1 * original_width / 640)
            y1 = int(y1 * original_height / 640)
            x2 = int(x2 * original_width / 640)
            y2 = int(y2 * original_height / 640)

            score = scores[i]
            class_id = class_ids[i]
            label = f'{class_names[class_id]}: {score:.2f}'
            animal_name = class_names[class_id]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Play sound for detected animal
            play_sound(animal_name)

    # Display the frame with YOLO predictions
    cv2.imshow('YOLO Predictions', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()