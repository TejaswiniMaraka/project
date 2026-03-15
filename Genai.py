import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import re

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🟢 Green Main Title
st.markdown(
    "<h1 style='color:green; text-align:center;'>AI Answer Evaluator</h1>",
    unsafe_allow_html=True
)

# Input method at top
st.subheader("Choose Input Method")
input_method = st.radio("", ["Type Text", "Upload PDF"])

model_answer = ""
student_answer = ""

# 🔵 Model Answer
st.markdown(
    "<h3 style='color:blue;'>Model Answer</h3>",
    unsafe_allow_html=True
)

if input_method == "Type Text":
    model_answer = st.text_area("Enter Model Answer Here", key="model_text")
else:
    model_file = st.file_uploader("Upload Model Answer PDF", type=["pdf"], key="model_file")
    if model_file is not None:
        pdf = fitz.open(stream=model_file.read(), filetype="pdf")
        for page in pdf:
            model_answer += page.get_text()
        st.success("Model Answer PDF Uploaded Successfully ✅")

# 💗 Student Answer
st.markdown(
    "<h3 style='color:pink;'>Student Answer</h3>",
    unsafe_allow_html=True
)

if input_method == "Type Text":
    student_answer = st.text_area("Enter Student Answer Here", key="student_text")
else:
    student_file = st.file_uploader("Upload Student Answer PDF", type=["pdf"], key="student_file")
    if student_file is not None:
        pdf = fitz.open(stream=student_file.read(), filetype="pdf")
        for page in pdf:
            student_answer += page.get_text()
        st.success("Student Answer PDF Uploaded Successfully ✅")


# --------- Evaluation ---------
if st.button("Evaluate"):
    if model_answer and student_answer:

        # 1️⃣ Semantic Similarity
        emb1 = model.encode([model_answer])
        emb2 = model.encode([student_answer])
        similarity = cosine_similarity(emb1, emb2)[0][0]

        # 2️⃣ Keyword Matching
        model_words = re.findall(r'\b\w+\b', model_answer.lower())
        student_words = re.findall(r'\b\w+\b', student_answer.lower())

        # Remove small words
        important_words = [word for word in model_words if len(word) > 3]

        if important_words:
            matched = sum(1 for word in important_words if word in student_words)
            keyword_score = matched / len(important_words)
        else:
            keyword_score = 0

        # 3️⃣ Final Combined Score
        final_score = (0.7 * similarity) + (0.3 * keyword_score)
        marks = round(final_score * 10, 2)

        # Result
        st.subheader("Result")
        st.write("Semantic Similarity:", round(similarity, 2))
        st.write("Keyword Match Score:", round(keyword_score, 2))
        st.write("Final Score:", round(final_score, 2))
        st.write("Marks (Out of 10):", marks)

        # Suggestions
        st.subheader("Suggestion")

        if final_score >= 0.85:
            st.success("Excellent Answer! 🌟")
        elif final_score >= 0.65:
            st.success("Very Good Answer 👍")
        elif final_score >= 0.45:
            st.info("Good Answer 🙂 But you can improve.")
        elif final_score >= 0.30:
            st.warning("Needs Improvement ⚠ Add more relevant keywords.")
        else:
            st.error("Poor Answer ❌ Incorrect or missing key concepts.")

    else:
        st.warning("Please provide both Model and Student answers.")