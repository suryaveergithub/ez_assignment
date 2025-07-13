from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
import spacy

class QAModel:
    def __init__(self, document_text):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Split document into clean sentences
        self.sentences = [sent.text.strip() for sent in self.nlp(document_text).sents if len(sent.text.strip()) > 10]

        # Create and store embeddings
        self.embeddings = self.embedder.encode(self.sentences, convert_to_numpy=True, normalize_embeddings=True)

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def answer(self, question, top_k=1):
        query_embedding = self.embedder.encode(question, convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(np.array([query_embedding]), top_k)
        return [self.sentences[i] for i in I[0]]
