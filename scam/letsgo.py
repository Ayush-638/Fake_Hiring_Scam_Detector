

import streamlit as st
import joblib
from io import BytesIO
import fitz  # PyMuPDF (for PDF)
from docx import Document  # python-docx (for DOCX)

# Load the trained model
model = joblib.load('model/fake_job_classifier.pkl')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Streamlit UI
st.title("Fake Job Offer Detector")
st.markdown("Upload a PDF or DOCX file to detect if it's a fake job offer.")

# File upload
uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file is not None:
    # Extract text based on file type
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)

    # Display extracted text (for user reference)
    st.subheader("Extracted Text")
    st.text_area("Text from your document", text, height=300)

    # Prediction
    if text:
        # Predict using the trained model
        prediction = model.predict([text])
        result = "Fake Job Offer" if prediction[0] == 1 else "Real Job Offer"
        st.subheader("Prediction: " + result)


