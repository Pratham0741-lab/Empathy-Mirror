import cv2
import threading
import speech_recognition as sr
from flask import Flask, render_template, Response, jsonify, send_file
from deepface import DeepFace
from textblob import TextBlob
import time
import datetime
import io

app = Flask(__name__)

# GLOBAL STATE
mirror_state = {
    "visual_emotion": "neutral",
    "emotion_spectrum": {},
    "current_transcript": "Waiting for speech...",
    "impact_label": "Ready",
    "advice": "System is listening. Speak clearly.",
    "history": [],
    "status": "Idle"
}

def audio_processor():
    recognizer = sr.Recognizer()
    
    # Settings for Real-Time Speech
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    
    # IF MIC IS NOT WORKING, PUT YOUR ID HERE (e.g., device_index=1)
    mic_id = None 
    
    with sr.Microphone(device_index=mic_id) as source:
        print("--- CALIBRATING BACKGROUND NOISE... ---")
        mirror_state["status"] = "Calibrating..."
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("--- LISTENING ---")
        mirror_state["status"] = "Listening"

        while True:
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=3)
                mirror_state["status"] = "Analyzing..."
                text = recognizer.recognize_google(audio)
                
                # Logic
                sentiment = TextBlob(text).sentiment.polarity
                visual = mirror_state["visual_emotion"]
                
                impact = "Neutral"
                advice = "Keep going."
                
                if sentiment > 0.1 and visual in ['happy', 'surprise']:
                    impact = "High Resonance"
                    advice = "Great energy!"
                elif sentiment > 0.1 and visual in ['sad', 'angry', 'fear']:
                    impact = "Mixed Signals"
                    advice = "Positive words, but tense face."
                elif sentiment < -0.1 and visual in ['happy']:
                    impact = "Masking"
                    advice = "You're smiling while saying something negative."

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
                mirror_state["status"] = "Listening"
                
            except:
                mirror_state["status"] = "Listening"
                continue

def video_processor():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        if int(time.time() * 10) % 5 == 0:
            try:
                res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
                
                # --- THE FIX IS HERE ---
                # We convert the numpy float32 values to standard python floats
                raw_emotions = res[0]['emotion']
                clean_emotions = {k: float(v) for k, v in raw_emotions.items()}
                
                mirror_state["emotion_spectrum"] = clean_emotions
                mirror_state["visual_emotion"] = res[0]['dominant_emotion']
            except: pass

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(video_processor(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data(): return jsonify(mirror_state)

@app.route('/download_report')
def download_report():
    output = io.BytesIO()
    report_text = f"EMPATHY MIRROR SESSION - {datetime.datetime.now()}\n"
    report_text += "="*50 + "\n\n"
    for entry in mirror_state["history"]:
        report_text += f"[{entry['time']}] {entry['text']}\n"
        report_text += f"   > Emotion: {entry['emotion']} | Impact: {entry['impact']}\n"
        report_text += "-"*30 + "\n"
    output.write(report_text.encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype="text/plain", as_attachment=True, download_name="session_history.txt")

if __name__ == '__main__':
    threading.Thread(target=audio_processor, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)