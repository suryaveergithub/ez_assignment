import streamlit as st
import fitz  # PyMuPDF
from summarizer import summarize_text
from qa_engine import QAModel
from quiz_engine import QuizModel

st.set_page_config(page_title="DocSmart AI", layout="wide")

# Dark/light mode-safe styling
st.markdown("""
    <style>
    html, body, .main {
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput > div > div > input {
        padding: 6px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    .card {
        padding: 1.25rem;
        border-radius: 6px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    .header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 DocSmart AI")
st.caption("AI-powered assistant for summarizing and understanding research documents")

uploaded_file = st.file_uploader("📤 Upload a PDF or TXT file", type=["pdf", "txt"])

def extract_text(file):
    if file.name.endswith('.pdf'):
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.name.endswith('.txt'):
        return file.read().decode("utf-8")
    return ""

if uploaded_file:
    text = extract_text(uploaded_file)
    summary = summarize_text(text)

    tab1, tab2, tab3 = st.tabs(["📝 Summary", "🤔 Ask Anything", "🧪 Challenge Me"])

    # Summary Tab
    with tab1:
        st.markdown("<div class='card'><div class='header'>Document Summary</div>" +
                    f"<div style='font-size: 1rem; color: inherit;'>{summary}</div></div>", unsafe_allow_html=True)

    # Ask Anything Tab
    with tab2:
        st.markdown("<div class='header'>Ask a question from this document</div>", unsafe_allow_html=True)
        question = st.text_input("Enter your question:")

        if "qa" not in st.session_state:
            st.session_state.qa = QAModel(text)

        if question:
            answer = st.session_state.qa.answer(question, top_k=1)[0]
            st.markdown(f"<div class='card'><div class='header'>Answer</div><div>{answer}</div></div>", unsafe_allow_html=True)

    # Challenge Me Tab
    with tab3:
        st.markdown("<div class='header'>Test your understanding of the document</div>", unsafe_allow_html=True)

        if "quiz" not in st.session_state:
            with st.spinner("🧪 Generating quiz..."):
                st.session_state.quiz = QuizModel(text)
                st.session_state.questions = st.session_state.quiz.get_questions()
                st.session_state.answers = [""] * len(st.session_state.questions)

        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state.questions):
                st.session_state.answers[i] = st.text_input(f"Q{i+1}: {q}", value=st.session_state.answers[i])
            submitted = st.form_submit_button("Submit Answers")

        if submitted:
            st.subheader("📝 Evaluation Results")
            for i, (q, ua) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
                result, reason = st.session_state.quiz.evaluate(q, ua)
                st.markdown(f"<div class='card'><div class='header'>Q{i+1}</div>"
                            f"<strong>Question:</strong> {q}<br>"
                            f"<strong>Your Answer:</strong> {ua}<br>"
                            f"<strong>Result:</strong> {result} <small style='opacity: 0.6;'>({reason})</small></div>", unsafe_allow_html=True)
