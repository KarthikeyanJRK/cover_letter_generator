import streamlit as st
from transformers import BartTokenizer, TFBartForConditionalGeneration
import pdfplumber
from pyresparser import ResumeParser
import requests
from bs4 import BeautifulSoup

# Streamlit UI setup
st.set_page_config(layout="wide")

# Custom CSS to inject for styling
st.markdown("""
<style>
body {
    color: #000;  /* Black text for better readability on light backgrounds */
    background-color: #111;
}
/* Apply styles to Streamlit components using their CSS class names */
.css-1e5imcs, .css-1cpxqw2, .st-cb {
    background-color: #fff; /* Light background for inputs and buttons */
    color: #000; /* Black text */
    border: 1px solid #fff; /* White border to blend with the background */
}

/* Adjust the button appearance */
.stButton > button {
    background-color: #000; /* Black background for buttons */
    color: #fff; /* White text */
    font-weight: bold; /* Bold font for buttons */
}

/* Simulate an A4 paper */
.a4-container {
    background: white;
    border: 2px solid #000;  /* Black border around the A4 container */
    border-radius: 5px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    height: 842px;  /* A4 size: 842px in height */
    width: 595px;   /* A4 size: 595px in width */
    overflow-y: auto;  /* Add scroll for text overflow */
    margin-left: auto;
    margin-right: auto; /* Center the A4 container within the column */
}

/* Bold labels */
label {
    font-weight: bold !important;
}

/* Remove Streamlit's default frame around the block elements */
.stTextInput, .stSelectbox, .stTextarea, .stFileUploader, .stButton, .css-1cpxqw2, .css-1e5imcs {
    border: none !important;
    box-shadow: none !important;
}

/* Input field adjustments */
input, textarea, select {
    border: 1px solid #ccc !important;  /* Slightly lighter border for input fields */
}

/* Fix for file uploader button to match the style */
input[type="file"] {
    color: transparent !important;
}

/* Adjust file uploader label */
.stFileUploader .css-8atqhb {
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)

# Load the BART model and tokenizer
def load_model():
    model_dir = 'KarthikeyanJRKIyer/cover_letter_generator'
    model = TFBartForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model()

# Function to extract text using pdfplumber
def extract_text_with_pdfplumber(pdf_file):
    text = []
    with pdfplumber.open(pdf_file) as pdf:
        pages = pdf.pages
        for page in pages:
            text += page.extract_text().split('\n')
    return text

# Function to find qualifications from the extracted text
def find_qualifications(text):
    qualifications = []
    keywords = ['university', 'bachelor', 'master', 'bsc', 'm.sc', 'phd', 'college', 'school', 'diploma']
    for line in text:
        if any(keyword in line.lower() for keyword in keywords):
            qualifications.append(line)
    return qualifications

# Function to parse resume PDF and extract data
def parse_resume(pdf_file):
    data = ResumeParser(pdf_file).get_extracted_data()
    all_text = extract_text_with_pdfplumber(pdf_file)
    qualifications = find_qualifications(all_text)
    data['qualification'] = qualifications
    return data

# Function to generate cover letter
def generate_cover_letter(input_data):
    inputs = tokenizer(input_data, return_tensors="tf", truncation=True, padding="max_length", max_length=512)
    output_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=512,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Function to extract job details from a URL
def extract_job_details(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        job_title_tag = soup.select_one('h1')
        job_title = job_title_tag.text.strip() if job_title_tag else 'Not found'

        next_a_tag = job_title_tag.find_next('a') if job_title_tag else None
        company_name = next_a_tag.text.strip() if next_a_tag else 'Not found'

        text = soup.get_text(separator=' ')
        sentences = text.split('.')
        relevant_qualifications = [sentence.strip() for sentence in sentences if any(term in sentence.lower() for term in ['experience with', 'experience of', 'years of'])]
        
        return {
            'Job Title': job_title,
            'Hiring Company': company_name,
            'Preferred Qualifications': relevant_qualifications
        }
    else:
        return {
            'Job Title': 'Not found',
            'Hiring Company': 'Not found',
            'Preferred Qualifications': []
        }

# Initializations before user inputs
job_title = ''
hiring_company = ''
preferred_qualifications = []
cover_letter_generated = None  # Ensure it's initialized

# Main content
col1, col2 = st.columns((1, 1))  # Adjust the ratio to your liking

with col1:
    st.title('Cover Letter Generator')
    job_url = st.text_input("Enter Job Posting URL")

    if job_url:
        job_details = extract_job_details(job_url)
        job_title = job_details['Job Title']
        hiring_company = job_details['Hiring Company']
        preferred_qualifications = job_details['Preferred Qualifications']

    job_title_area = st.text_input("Job Title:", value=job_title)
    hiring_company_area = st.text_input("Hiring Company:", value=hiring_company)
    preferred_qualifications_area = st.text_area("Preferred Qualifications:", value=', '.join(preferred_qualifications), height=100)

    uploaded_file = st.file_uploader("Upload Resume", type=['pdf'])

    if uploaded_file is not None:
        resume_data = parse_resume(uploaded_file)
        applicant_name = resume_data.get('name', '')
        past_experience = ' '.join(resume_data.get('experience', []))
        skills = ', '.join(resume_data.get('skills', []))
        qualifications_list = resume_data.get('qualification', [])
    else:
        applicant_name = ''
        past_experience = ''
        skills = ''
        qualifications_list = []

    applicant_name = st.text_input("Applicant's Name", value=applicant_name)
    past_experience = st.text_input('Past Working Experience', value=past_experience)
    skillsets = st.text_input('Skillsets', value=skills)
    qualifications = st.text_area('Qualifications', value=' '.join(qualifications_list), height=100)

    if st.button('Generate Cover Letter'):
        input_data = f"Job Title: {job_title_area}, Hiring Company: {hiring_company_area}, " \
                    f"Applicant Name: {applicant_name}, Past Working Experience: {past_experience}, " \
                    f"Skillsets: {skillsets}, Qualifications: {qualifications}, " \
                    f"Preferred Qualifications: {preferred_qualifications_area}"
        cover_letter_generated = generate_cover_letter(input_data)
        cover_letter_generated = cover_letter_generated.replace("Dear Hiring Manager,", "Dear Hiring Manager,\n\n")  # Ensure newlines are respected
        cover_letter_generated = cover_letter_generated.replace("Thank you for considering", "\n\nThank you for considering")
        cover_letter_generated = cover_letter_generated.replace("Sincerely,", "\n\nSincerely,\n\n")

with col2:
    #st.markdown(a4_css, unsafe_allow_html=True)
    if cover_letter_generated:
        # Display the cover letter inside a div that simulates an A4 paper sheet
        st.markdown(f'<div class="a4-container">{cover_letter_generated}</div>', unsafe_allow_html=True)
    else:
        st.markdown("## Instructions")
        st.markdown("""
        - **Step 1:** Enter the job posting URL in the field on the left.
        - **Step 2:** The job title, company, and preferred qualifications will be fetched automatically.
        - **Step 3:** Upload your resume in PDF format.
        - **Step 4:** Verify and edit your details such as name, past experience, skillsets, and qualifications.
        - **Step 5:** Click the 'Generate Cover Letter' button.
        - **Step 6:** Your personalized cover letter will appear here.
        """)
        st.markdown("### Additional Tips")
        st.markdown("""
        - Ensure that the job posting URL is correct to fetch accurate details.
        - Customize the generated cover letter further if needed to add a personal touch.
        - Double-check for any typos or errors in the final cover letter before using it.
        """)

# Ensure all the functions are included as per your original code and any additional logic you require.
