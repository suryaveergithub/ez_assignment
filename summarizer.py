import re
import heapq
import spacy

# Load SpaCy model once
nlp = spacy.load("en_core_web_sm")

def summarize_text(text, max_sentences=5):
    # Clean text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', '', text)

    # Use SpaCy for tokenization
    doc = nlp(text)
    word_freq = {}

    for token in doc:
        if token.is_alpha and not token.is_stop:
            word = token.text.lower()
            word_freq[word] = word_freq.get(word, 0) + 1

    max_freq = max(word_freq.values(), default=1)
    for word in word_freq:
        word_freq[word] /= max_freq

    # Score sentences
    sentence_scores = {}
    for sent in doc.sents:
        score = 0
        for word in sent:
            if word.text.lower() in word_freq:
                score += word_freq[word.text.lower()]
        sentence_scores[sent.text] = score

    # Top N sentences
    summary_sentences = heapq.nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)
