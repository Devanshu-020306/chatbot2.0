# kb_store.py
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from faq_data import faq_pairs
from typing import Tuple
import re

DB_PATH = "kb_data.db"

class KBStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._ensure_db()

        # Load FAQ data
        self.faqs = faq_pairs
        self.corpus_questions = [self.clean_text(f["question"]) for f in self.faqs]

        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_questions)

    def _ensure_db(self):
        init_needed = not os.path.exists(self.db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        if init_needed:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT,
                    user_message TEXT,
                    bot_reply TEXT,
                    score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT,
                    reminder_text TEXT,
                    remind_at TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def find_most_similar(self, text: str) -> Tuple[int, float]:
        text_clean = self.clean_text(text)
        q_vec = self.vectorizer.transform([text_clean])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        return best_idx, best_score

    def get_answer_by_index(self, idx: int) -> str:
        if 0 <= idx < len(self.faqs):
            return self.faqs[idx]["answer"]
        return "Sorry, I don't have an answer for that."

    def log_interaction(self, user_name: str, user_message: str, bot_reply: str, score: float):
        self.cursor.execute(
            "INSERT INTO logs (user_name, user_message, bot_reply, score) VALUES (?, ?, ?, ?)",
            (user_name, user_message, bot_reply, score)
        )
        self.conn.commit()

    def add_reminder(self, user_name: str, reminder_text: str, remind_at: str):
        self.cursor.execute(
            "INSERT INTO reminders (user_name, reminder_text, remind_at) VALUES (?, ?, ?)",
            (user_name, reminder_text, remind_at)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

# Singleton KBStore instance
kb_store = KBStore()
SIMILARITY_THRESHOLD = 0.3  # Adjust based on testing
