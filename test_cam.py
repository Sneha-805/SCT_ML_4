import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not working")
else:
    print("✅ Webcam working")
    cap.release()
