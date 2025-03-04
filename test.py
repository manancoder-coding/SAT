import streamlit as st
import fitz  # PyMuPDF
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API key securely from Streamlit secrets
openai_client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")  # Extract text from each page
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    """Chunks text into manageable parts."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

def analyze_performance(questions, answers, responses):
    """Analyzes user performance and provides insights using LLM."""

    max_length = 4000  # Reduce this if needed

    # Truncate input if it's too long
    questions = questions[:max_length]
    answers = answers[:max_length]
    responses = responses[:max_length]

    prompt = f"""
    Analyze the following SAT test performance:
    Questions: {questions}
    Correct Answers: {answers}
    User Responses: {responses}
    Identify weak topics, strong topics, and suggest a 1-week study plan.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are an AI SAT tutor analyzing student performance."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Streamlit App UI
st.title("AI-Powered SAT Preparation Pipeline")

st.header("Upload Study Materials")
study_materials = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if study_materials:
    extracted_text = ""
    for material in study_materials:
        extracted_text += extract_text_from_pdf(material)
    chunked_materials = chunk_text(extracted_text)
    st.success("Study materials processed successfully!")

st.header("Upload SAT Test Data")
questions_file = st.file_uploader("Upload SAT Test Questions", type=["pdf"])
answers_file = st.file_uploader("Upload Correct Answers", type=["pdf"])
responses_file = st.file_uploader("Upload Your Responses", type=["pdf"])

if questions_file and answers_file and responses_file:
    questions = extract_text_from_pdf(questions_file)
    answers = extract_text_from_pdf(answers_file)
    responses = extract_text_from_pdf(responses_file)
    
    st.success("Test data uploaded successfully!")
    
    if st.button("Generate Analysis Report"):
        report = analyze_performance(questions, answers, responses)
        st.subheader("Your Performance Analysis Report")
        st.write(report)
        st.download_button("Download Report", report, "sat_report.txt")
