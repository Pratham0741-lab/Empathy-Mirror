import cv2
import threading
import numpy as np
import json
import time
import datetime
import io
import logging
import pyaudio
from flask import Flask, render_template, Response, jsonify, send_file
from flask.json.provider import DefaultJSONProvider
from deepface import DeepFace
from textblob import TextBlob
from vosk import Model, KaldiRecognizer

# --- CONFIGURATION ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# JSON Fix
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
app.json = NumpyJSONProvider(app)

# --- STATE ---
mirror_state = {
    "visual_emotion": "neutral",
    "emotion_spectrum": {},
    "current_transcript": "Ready. Speak naturally.",
    "impact_label": "Ready",
    "advice": "System is offline & ready.",
    "history": [],
    "status": "Idle",
    "session_start": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
}

# --- OFFLINE AUDIO ENGINE (VOSK) ---
def audio_loop():
    # Load Model (Ensure 'model' folder exists!)
    try:
        print(">> LOADING OFFLINE MODEL... (This may take 5 seconds)")
        model = Model("model")
        rec = KaldiRecognizer(model, 16000)
        print(">> MODEL LOADED. MIC ACTIVE.")
    except Exception as e:
        print(f"ERROR: Could not load model. Make sure 'model' folder exists.\n{e}")
        return

    # PyAudio Setup
    p = pyaudio.PyAudio()
    
    # Try to find a working input device
    input_device_index = None
    # Uncomment lines below to debug mics if needed:
    # for i in range(p.get_device_count()):
    #     print(p.get_device_info_by_index(i))

    stream = p.open(format=pyaudio.paInt16, 
                    channels=1, 
                    rate=16000, 
                    input=True, 
                    frames_per_buffer=4000,
                    input_device_index=input_device_index)
    
    stream.start_stream()
    mirror_state["status"] = "Active"

    while True:
        try:
            # Read chunk of audio
            data = stream.read(4000, exception_on_overflow=False)
            
            if rec.AcceptWaveform(data):
                # Full sentence detected
                result = json.loads(rec.Result())
                text = result.get("text", "")
                
                if text:
                    process_transcript(text)
            else:
                # Partial result (Real-time updates while you speak!)
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    mirror_state["current_transcript"] = partial_text + "..."

        except Exception as e:
            print(f"Audio Error: {e}")
            continue

def process_transcript(text):
    # Logic Engine
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    visual = mirror_state["visual_emotion"]
    
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

# --- VIDEO ENGINE ---
def video_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Analyze every 4th frame
        if int(time.time() * 100) % 4 == 0:
            try:
                result = DeepFace.analyze(
                    frame, actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv', 
                    silent=True
                )
                mirror_state["emotion_spectrum"] = result[0]['emotion']
                mirror_state["visual_emotion"] = result[0]['dominant_emotion']
            except: pass

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(video_loop(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data(): return jsonify(mirror_state)

@app.route('/download')
def download():
    output = io.BytesIO()
    txt = f"SESSION REPORT - {mirror_state['session_start']}\n{'='*50}\n"
    for item in mirror_state["history"]:
        txt += f"[{item['time']}] {item['text']}\n"
        txt += f"    > Face: {item['emotion']} | Impact: {item['impact']}\n{'-'*30}\n"
    output.write(txt.encode('utf-8'))
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="report.txt", mimetype="text/plain")

if __name__ == '__main__':
    threading.Thread(target=audio_loop, daemon=True).start()
    from waitress import serve
    print("SERVER STARTED: http://localhost:8080")
    serve(app, host='0.0.0.0', port=8080, threads=6)