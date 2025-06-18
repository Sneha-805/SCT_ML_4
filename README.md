# 🖐️ Gesture Recognition using CNN | SkillCraft Internship Task 4
This project is built as part of Task 4 of my internship at SkillCraft Technology. It involves building a Gesture Recognition System using Convolutional Neural Networks (CNN) with a real-time webcam interface powered by Flask.

## 📌 Features
🧠 CNN-based gesture recognition model

🎥 Real-time webcam feed using OpenCV

🌐 Flask-based web interface

💾 Trained model saved as .h5 file

🖥️ Predicts hand gesture classes in live video

# 🛠️ Tech Stack
Python

TensorFlow / Keras

OpenCV

Flask

HTML / CSS

NumPy

# 📂 Project Structure
bash
Copy
Edit
gesture-flask-app/
│
├── gesture_train.py         # CNN model training script
├── app.py                   # Flask app for real-time webcam gesture recognition
├── gesture_model.h5         # Trained gesture recognition model
├── templates/
│   └── index.html           # Frontend page for Flask
└── static/
    └── style.css            # Styling for the web page (optional)
# 🧪 How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/Sneha-805/gesture-flask-app.git
cd gesture-flask-app
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Train the model

bash
Copy
Edit
python3 gesture_train.py
Run the Flask app

bash
Copy
Edit
python3 app.py
Open in browser

cpp
Copy
Edit
http://127.0.0.1:5000
# 🚀 Output
Displays "Webcam Working" once the camera feed is initialized.

Recognizes hand gestures trained by the CNN model in real time.

# 📦 Requirements
Create a requirements.txt with:

txt
Copy
Edit
flask
opencv-python
tensorflow
numpy

## 👩‍💻 Author
Sneha Mudda
B.Tech CSE, IIIT RK Valley
[LinkedIn](https://www.linkedin.com/in/sneha-mudda-b57819282/) | [GitHub](https://github.com/Sneha-805/)
