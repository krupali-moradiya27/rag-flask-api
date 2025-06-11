# app.py
from flask import Flask, request, jsonify
from generate_response import generate_response
import os

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Server is running"}), 200

@app.route("/rag-chat", methods=["POST"])
def rag_chat():
    # 🔒 Step 1: Validate API Key
    request_key = request.headers.get("x-api-key")
    print("hello")
    if request_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # 🔁 Step 2: Process request
    data = request.get_json()
    user_query = data.get("query")
    health_data = data.get("health_data", {})
    language = data.get("language", "en")

    if not user_query:
        return jsonify({"error": "Missing query"}), 400

    try:
        response = generate_response(user_query, health_data, language)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



