import streamlit as st
from transformers import BartTokenizer, TFBartForConditionalGeneration
import pdfplumber
from pyresparser import ResumeParser
import requests
from bs4 import BeautifulSoup

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


# Streamlit UI setup
st.title('Cover Letter Generator')
job_url = st.text_input("Enter Job Posting URL")

# Fetch job details if URL is entered
job_details = {}
if job_url:
    job_details = extract_job_details(job_url)
    job_title = job_details.get('Job Title', 'Not found')
    hiring_company = job_details.get('Hiring Company', 'Not found')
    preferred_qualifications = ', '.join(job_details.get('Preferred Qualifications', []))
else:
    job_title = ''
    hiring_company = ''
    preferred_qualifications = ''

# Display fetched job details
st.write("Job Title:", job_title)
st.write("Hiring Company:", hiring_company)
st.write("Preferred Qualifications:", preferred_qualifications)

uploaded_file = st.file_uploader("Upload Resume", type=['pdf'])

if uploaded_file is not None:
    resume_data = parse_resume(uploaded_file)
    applicant_name = resume_data.get('name', '')
    past_experience = ' '.join(resume_data.get('experience', []))
    skills = ', '.join(resume_data.get('skills', []))
    qualifications = ' '.join(resume_data.get('qualification', []))
else:
    applicant_name = ''
    past_experience = ''
    skills = ''
    qualifications = ''

applicant_name = st.text_input("Applicant's Name", value=applicant_name)
past_experience = st.text_input('Past Working Experience', value=past_experience)
current_experience = st.text_input('Current Working Experience')
skillsets = st.text_input('Skillsets', value=skills)
qualifications = st.text_input('Qualifications', value=qualifications)

if st.button('Generate Cover Letter'):
    input_data = f"Job Title: {job_title}, Preferred Qualifications: {preferred_qualifications}, Hiring Company: {hiring_company}, Applicant Name: {applicant_name}, Past Working Experience: {past_experience}, Current Working Experience: {current_experience}, Skillsets: {skillsets}, Qualifications: {qualifications}"
    cover_letter = generate_cover_letter(input_data)
    cover_letter = cover_letter.replace("Dear Hiring Manager,", "Dear Hiring Manager,\n")
    cover_letter = cover_letter.replace("Thank you for considering", "\nThank you for considering")
    cover_letter = cover_letter.replace("Sincerely,", "\nSincerely,")
    st.text_area("Cover Letter:", cover_letter, height=300)
