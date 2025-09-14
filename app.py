from flask import Flask, request, jsonify, render_template
from kb_store import kb_store  # make sure kb_store.py exists and has kb_store instance

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_message = data.get("message", "")
    user_name = data.get("user_name", "Guest")

    # Get the most similar FAQ answer
    idx, score = kb_store.find_most_similar(user_message)
    bot_reply = kb_store.get_answer_by_index(idx)

    # Log interaction
    kb_store.log_interaction(user_name, user_message, bot_reply, score)

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
