from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load model and labels
model = load_model('model/emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Global state
cap = None
streaming = False

# Emotion tracking
emotion_durations = {emotion: 0 for emotion in emotion_labels}
current_emotion = None
start_time = None

def reset_emotion_data():
    global emotion_durations, current_emotion, start_time
    emotion_durations = {emotion: 0 for emotion in emotion_labels}
    current_emotion = None
    start_time = None

def update_emotion_duration(new_emotion):
    global current_emotion, start_time, emotion_durations
    current_time = time.time()

    if current_emotion is not None and start_time is not None:
        elapsed = current_time - start_time
        emotion_durations[current_emotion] += elapsed

    current_emotion = new_emotion
    start_time = current_time

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return frame

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_array = img_to_array(roi_resized) / 255.0
        roi_array = np.expand_dims(roi_array, axis=0)

        prediction = model.predict(roi_array, verbose=0)
        emotion_index = np.argmax(prediction[0])
        emotion = emotion_labels[emotion_index]

        update_emotion_duration(emotion)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

def generate_frames():
    global cap, streaming
    cap = cv2.VideoCapture(0)
    while streaming:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    if cap:
        cap.release()

def plot_emotion_duration():
    fig, ax = plt.subplots(figsize=(10, 5))
    emotions = list(emotion_durations.keys())
    durations = list(emotion_durations.values())
    ax.bar(emotions, durations, color='skyblue')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Time Spent on Each Emotion')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return encoded

@app.route('/')
def index():
    return render_template('emotion.html')

@app.route('/video_feed')
def video_feed():
    global streaming
    reset_emotion_data()
    streaming = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global streaming
    streaming = False
    return ('', 204)

@app.route('/get_graph')
def get_graph():
    graph_base64 = plot_emotion_duration()
    return f"data:image/png;base64,{graph_base64}"

if __name__ == "__main__":
    app.run()
