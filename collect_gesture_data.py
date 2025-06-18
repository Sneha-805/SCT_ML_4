import cv2
import os

# === CONFIGURATION ===
gestures = ["thumbs_up", "stop", "peace"]  # You can add more here
images_per_gesture = 200
output_dir = "gesture_dataset"
capture_size = 224  # image size (width & height)

# === CREATE FOLDERS ===
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for gesture in gestures:
    os.makedirs(os.path.join(output_dir, gesture), exist_ok=True)

# === START CAPTURING ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not found.")
    exit()

for gesture in gestures:
    print(f"\n📸 Get ready to record: {gesture}")
    print("Press 's' to start recording...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Gesture: {gesture} - Press 's' to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
        cv2.imshow("Capture Gesture", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    count = 0
    while count < images_per_gesture:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        roi = cv2.resize(frame, (capture_size, capture_size))
        path = os.path.join(output_dir, gesture, f"{count}.jpg")
        cv2.imwrite(path, roi)
        count += 1

        cv2.putText(frame, f"Capturing {gesture}: {count}/{images_per_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("\n✅ Dataset collection complete!")
cap.release()
cv2.destroyAllWindows()
