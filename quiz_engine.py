import random
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

class QuizModel:
    def __init__(self, text):
        self.text = text
        self.spacy_model = spacy.load("en_core_web_sm")
        self.sentences = [sent.text.strip() for sent in self.spacy_model(text).sents if len(sent.text.strip()) > 30]
        self.quiz_data = self._generate_fill_in_blank_questions(3)

        # for scoring
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(self.sentences, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def _generate_fill_in_blank_questions(self, num_qs):
        questions = []
        for sent in random.sample(self.sentences, k=min(num_qs, len(self.sentences))):
            doc = self.spacy_model(sent)
            keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 3]
            if keywords:
                answer = random.choice(keywords)
                q_text = sent.replace(answer, "_____")
                questions.append({"question": q_text, "answer": answer})
        return questions

    def get_questions(self):
        return [q["question"] for q in self.quiz_data]

    def evaluate(self, question, user_answer):
        matched = next((q for q in self.quiz_data if q["question"] == question), None)
        if not matched:
            return "❓ Question not found", "N/A"

        ref_answer = matched["answer"]
        score = float(util.cos_sim(
            self.embedder.encode(user_answer, convert_to_numpy=True, normalize_embeddings=True),
            self.embedder.encode(ref_answer, convert_to_numpy=True, normalize_embeddings=True)
        ))

        result = "✅ Correct" if score > 0.6 else "❌ Needs Improvement"
        return result, f"Expected: {ref_answer} | Similarity: {score:.2f}"
