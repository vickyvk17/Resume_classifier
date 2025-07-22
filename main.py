import streamlit as st
import pandas as pd
import joblib
import re
from PyPDF2 import PdfReader
import docx2txt
import os

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    return docx2txt.process(file)

# Streamlit UI
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("🧠 Resume Classifier")
st.markdown("Upload a resume, and I'll predict the job role (like Data Scientist, Web Developer, etc.)")

uploaded_file = st.file_uploader("📁 Upload Resume (PDF, DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    # Extract text based on file type
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

    if file_extension == ".pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == ".docx":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or DOCX.")
        st.stop()

    if not resume_text.strip():
        st.warning("Couldn't extract any text from the file.")
        st.stop()

    # Show raw resume text (optional)
    with st.expander("📄 Show Extracted Resume Text"):
        st.write(resume_text[:3000])  # limit to avoid overload

    # Clean and predict
    cleaned = clean_text(resume_text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]

    st.success(f"✅ **Predicted Job Role:** {prediction}")
