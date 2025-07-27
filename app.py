from flask import Flask, render_template, request, jsonify, session
import requests
import uuid
import os

from db import init_db, save_message, get_history, get_all_sessions, reset_session

app = Flask(__name__)
app.secret_key = os.urandom(24)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "turkcell-custom"

def generate_response(message, history):
    prompt = ""
    for item in history:
        prompt += f"Kullanıcı: {item['user']}\nAsistan: {item['bot']}\n"
    prompt += f"Kullanıcı: {message}\nAsistan:"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "").strip()
    else:
        return "⚠️ Sunucu hatası: Yanıt alınamadı."

@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    sid = session['session_id']
    history = get_history(sid)
    all_sessions = get_all_sessions()
    return render_template("chat.html", history=history, all_sessions=all_sessions)

@app.route("/send_message", methods=["POST"])
def send_message():
    user_msg = request.json.get("message")
    sid = session.get('session_id')

    history = get_history(sid)
    bot_msg = generate_response(user_msg, history)

    save_message(sid, user_msg, bot_msg)
    return jsonify({"response": bot_msg})

@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get('session_id')
    reset_session(sid)
    return jsonify({"status": "ok"})

@app.route("/switch_session", methods=["POST"])
def switch_session():
    new_sid = request.json.get("session_id")
    session['session_id'] = new_sid
    return jsonify({"status": "switched"})


@app.route("/train_model", methods=["GET", "POST"])
def train_model():
    if request.method == "POST":
        # Burada fine-tuning işlemi yapılabilir
        dataset = request.files.get("dataset")
        model_files = request.files.getlist("model_files")
        epochs = request.form.get("epochs")
        batch_size = request.form.get("batch_size")
        
        # 💡 Dosyaları ve parametreleri kontrol etmek için log at
        print("✅ Fine-tune başlatıldı")
        print("Dataset:", dataset.filename)
        print("Epochs:", epochs)
        print("Batch Size:", batch_size)
        print("Toplam model dosyası:", len(model_files))

        # Burada eğitim işlemini başlatabilirsin

        return "🚀 Model eğitimi başlatıldı!"
    
    # Eğer GET isteği gelirse, formu göster
    return render_template("train_model.html")

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
