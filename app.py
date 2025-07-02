# app.py
import pickle, numpy as np
from flask import Flask, request, jsonify, render_template
from model.utils import text_preprocess, calculate_tf, build_tf_idf, expit

# Load artifacts once
with open("model/artifacts/w.pkl", "rb") as f: w = pickle.load(f)
with open("model/artifacts/b.pkl", "rb") as f: b = pickle.load(f)
with open("model/artifacts/vocab_id.pkl", "rb") as f: vocab_id = pickle.load(f)
with open("model/artifacts/vocab.pkl", "rb") as f: vocab = pickle.load(f)
with open("model/artifacts/total_docs.pkl", "rb") as f: total_docs = pickle.load(f)


# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text","")
    tokens = text_preprocess(text)
    tf = calculate_tf([tokens])
    x = build_tf_idf(tf, vocab, vocab_id, total_docs)  # shape (1, D)
    y_hat = expit(np.dot(x, w.T) + b)[0]
    label = "Spam!" if y_hat > 0.5 else "Looks Good!"
    confidence = max(y_hat, 1 - y_hat)
    return jsonify({"label": label, "confidence": float(confidence)})

if __name__ == "__main__":
    app.run(debug=True)
