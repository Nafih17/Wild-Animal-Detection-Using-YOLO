import cv2
from ultralytics import YOLO
from playsound import playsound
import os
import threading

# Sound file paths for animals (use raw strings for paths)
sound_files = {
    "elephant": r"C:\Users\Admin\PycharmProjects\PythonProject1\2sounds_animal\2sounds_animal\dentist_drill.wav",
    "tiger": r"C:\Users\Admin\PycharmProjects\PythonProject1\2sounds_animal\2sounds_animal\construction.wav",
    "cheetah": r"C:\Users\Admin\PycharmProjects\PythonProject1\2sounds_animal\2sounds_animal\alarm_beep.wav",
    "lion": r"C:\Users\Admin\PycharmProjects\PythonProject1\2sounds_animal\2sounds_animal\air_raid.wav"
}

# Function to play sound in a separate thread
def play_sound(animal_name):
    if animal_name in sound_files:
        sound_file = sound_files[animal_name]
        if os.path.exists(sound_file):
            try:
                # Use threading to play sound asynchronously
                sound_thread = threading.Thread(target=playsound, args=(sound_file,))
                sound_thread.start()
            except Exception as e:
                print(f"Error playing sound for {animal_name}: {e}")
        else:
            print(f"Error: Sound file {sound_file} not found.")
    else:
        print(f"Error: No sound file available for {animal_name}")

# Path to the image and YOLO model
image_path = r"C:\Users\Admin\PycharmProjects\PythonProject1\dataset\dataset\train\images\IMG-20250102-WA0009.jpg"  # Change this to your image path
model_path = r"C:\Users\Admin\PycharmProjects\PythonProject1\final_best.pt"

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

# Load YOLO model
model = YOLO(model_path)

# Read the image
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image from {image_path}")
    exit()

# Predict objects in the image using YOLO
results = model.predict(img)

# Process the results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
    class_names = result.names  # Class names dictionary

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        score = scores[i]
        class_id = class_ids[i]
        label = f'{class_names[class_id]}: {score:.2f}'
        animal_name = class_names[class_id]

        # Draw bounding box and label on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Play sound for detected animal
        play_sound(animal_name)

# Display the image with YOLO predictions
cv2.imshow('YOLO Predictions', img)

# Wait until a key is pressed
cv2.waitKey(0)

# Close the image window
cv2.destroyAllWindows()
