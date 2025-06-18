from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

# Load the trained gesture recognition model
model = tf.keras.models.load_model('gesture_model.h5')

# Define class labels (Update based on your dataset)
classes = ['Thumbs Up', 'Thumbs Down', 'Ok', 'Peace', 'Fist']

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(2)

def preprocess_frame(frame):
    # Extract the region of interest (can adjust as per your setup)
    roi = frame[100:300, 100:300]
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Preprocess and predict
            roi = preprocess_frame(frame)
            pred = model.predict(roi)
            class_id = np.argmax(pred)
            class_label = classes[class_id]

            # Draw box and label
            cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
            cv2.putText(frame, class_label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield to the stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
