import cv2
import threading
import speech_recognition as sr
import numpy as np
import json
import time
import datetime
import io
import logging
from flask import Flask, render_template, Response, jsonify, send_file, request
from flask.json.provider import DefaultJSONProvider
from deepface import DeepFace
from textblob import TextBlob

# --- 1. SETUP & CONFIG ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- 2. JSON FIX (Prevents Crashes) ---
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json = NumpyJSONProvider(app)

# --- 3. STATE ---
mirror_state = {
    "visual_emotion": "neutral",
    "emotion_spectrum": {},
    "current_transcript": "Ready.",
    "impact_label": "Ready",
    "advice": "System ready.",
    "history": [],
    "status": "Idle",
    "session_start": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
}

# --- 4. FAST AUDIO ENGINE ---
def audio_loop():
    recognizer = sr.Recognizer()
    
    # LATENCY OPTIMIZATION SETTINGS
    recognizer.energy_threshold = 300      # Fixed threshold is faster than dynamic
    recognizer.dynamic_energy_threshold = False 
    recognizer.pause_threshold = 0.4       # Stop listening after 0.4s of silence (Very Fast)
    recognizer.non_speaking_duration = 0.3 # Buffer size
    
    # mic_id = 1  <-- UNCOMMENT AND SET THIS IF MIC ISN'T WORKING
    mic_id = None 

    with sr.Microphone(device_index=mic_id) as source:
        print(">> AUDIO: Calibrating (0.5s)...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        mirror_state["status"] = "Active"
        print(">> AUDIO: Started.")

        while True:
            try:
                # phrase_time_limit=2 forces an update every 2 seconds
                # This makes the transcript feel "streaming" rather than blocked
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=2)
                
                try:
                    text = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    continue # Skip empty audio to save processing time
                
                # Instant Logic
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                visual = mirror_state["visual_emotion"]
                
                # Impact Calculation
                impact = "Neutral"
                advice = "Keep flowing."
                
                if sentiment > 0.1 and visual in ['happy', 'surprise']:
                    impact = "High Resonance"
                    advice = "Great energy match!"
                elif sentiment > 0.1 and visual in ['sad', 'angry', 'fear']:
                    impact = "Mixed Signals"
                    advice = "Positive words, tense face."
                elif sentiment < -0.1 and visual in ['happy']:
                    impact = "Masking"
                    advice = "Smiling through negative news."
                elif visual == 'neutral':
                    impact = "Steady"
                    advice = "Add emotion to emphasize points."

                # Update State
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                entry = {
                    "time": timestamp,
                    "text": text,
                    "emotion": visual,
                    "impact": impact
                }
                
                mirror_state["current_transcript"] = text
                mirror_state["impact_label"] = impact
                mirror_state["advice"] = advice
                mirror_state["history"].insert(0, entry)
                
            except Exception as e:
                print(f"Audio Error: {e}")
                time.sleep(0.1)

# --- 5. VIDEO ENGINE ---
def video_loop():
    cap = cv2.VideoCapture(0)
    # Low Resolution = Low Latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Analyze every 4th frame (approx 7.5fps analysis)
        if int(time.time() * 100) % 4 == 0:
            try:
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv', 
                    silent=True
                )
                mirror_state["emotion_spectrum"] = result[0]['emotion']
                mirror_state["visual_emotion"] = result[0]['dominant_emotion']
            except: pass

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 6. ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(video_loop(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data(): return jsonify(mirror_state)

@app.route('/download')
def download():
    output = io.BytesIO()
    s = mirror_state
    txt = f"SESSION REPORT - {s['session_start']}\n{'='*50}\n"
    for item in s["history"]:
        txt += f"[{item['time']}] {item['text']}\n"
        txt += f"    > Face: {item['emotion']} | Impact: {item['impact']}\n{'-'*30}\n"
    output.write(txt.encode('utf-8'))
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="report.txt", mimetype="text/plain")

if __name__ == '__main__':
    threading.Thread(target=audio_loop, daemon=True).start()
    from waitress import serve
    print("---------------------------------------")
    print(" SERVER STARTED: http://localhost:8080")
    print("---------------------------------------")
    serve(app, host='0.0.0.0', port=8080, threads=6)