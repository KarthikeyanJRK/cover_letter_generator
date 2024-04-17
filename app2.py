import streamlit as st
from transformers import BartTokenizer, TFBartForConditionalGeneration
import fitz  # PyMuPDF

def load_model():
    model_dir = 'C://Users//Malhan//Downloads//Bart_Model//bart_model//content//bart_model'
    model = TFBartForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model()

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

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read())
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_info_from_text(text):
    # Splitting the resume text into lines
    lines = text.split('\n')
    info = {
        'applicant_name': '',
        'past_experience': '',
        'current_experience': '',
        'skillsets': '',
        'qualifications': ''
    }
    current_section = None

    for line in lines:
        line = line.strip()
        if 'Name' in line:
            info['applicant_name'] = line.split(':')[-1].strip()
        if 'Experience' in line or 'EXPERIENCE' in line:
            current_section = 'experience'
        elif 'Skills' in line or 'SKILLS' in line:
            current_section = 'skills'
        elif 'Education' in line or 'EDUCATION' in line or 'Qualifications' in line or 'QUALIFICATIONS' in line:
            current_section = 'qualifications'
        elif line.strip() == "":
            current_section = None

        if current_section == 'experience' and 'Experience' not in line and 'EXPERIENCE' not in line:
            if 'current' in line.lower() or 'present' in line.lower():
                info['current_experience'] += line + ' '
            else:
                info['past_experience'] += line + ' '
        elif current_section == 'skills':
            info['skillsets'] += line + ' '
        elif current_section == 'qualifications':
            info['qualifications'] += line + ' '

    return info


st.title('Cover Letter Generator')

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    extracted_info = extract_info_from_text(text)
    applicant_name = extracted_info.get('applicant_name', '')

job_title = st.text_input('Job Title')
preferred_qualifications = st.text_input('Preferred Qualifications')
hiring_company = st.text_input('Hiring Company')
applicant_name = st.text_input("Applicant's Name", value=applicant_name)
past_experience = st.text_input('Past Working Experience')
current_experience = st.text_input('Current Working Experience')
skillsets = st.text_input('Skillsets')
qualifications = st.text_input('Qualifications')

if st.button('Generate Cover Letter'):
    input_data = f"Job Title: {job_title}, Preferred Qualifications: {preferred_qualifications}, Hiring Company: {hiring_company}, Applicant Name: {applicant_name}, Past Working Experience: {past_experience}, Current Working Experience: {current_experience}, Skillsets: {skillsets}, Qualifications: {qualifications}"
    cover_letter = generate_cover_letter(input_data)
    st.text_area("Cover Letter:", cover_letter, height=300)
