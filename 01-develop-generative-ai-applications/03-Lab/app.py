from flask import Flask, request, jsonify, render_template
import time

from config import settings
from model import glm5_response, minimax_response, qwen_response

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    user_message = data.get("message")
    model = data.get("model")

    if not user_message or not model:
        return jsonify({"error": "Missing message or model selection"}), 400

    system_prompt = settings.system_prompt
    start_time = time.time()

    try:
        if model == "glm5":
            result = glm5_response(system_prompt, user_message)
        elif model == "minimax":
            result = minimax_response(system_prompt, user_message)
        elif model == "qwen":
            result = qwen_response(system_prompt, user_message)
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        result["duration"] = time.time() - start_time
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
