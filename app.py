pip install transformers
import streamlit as st
from transformers import BartTokenizer, TFBartForConditionalGeneration

def load_model():
    model_dir = 'C://Users//Malhan//Downloads//Bart_Model//bart_model//content//bart_model'
    model = TFBartForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model()

# Define the generate_cover_letter function
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

# Streamlit interface
st.title('Cover Letter Generator')

# Input fields
job_title = st.text_input('Job Title')
preferred_qualifications = st.text_input('Preferred Qualifications')
hiring_company = st.text_input('Hiring Company')
applicant_name = st.text_input("Applicant's Name")
past_experience = st.text_input('Past Working Experience')
current_experience = st.text_input('Current Working Experience')
skillsets = st.text_input('Skillsets')
qualifications = st.text_input('Qualifications')

# Generate button
if st.button('Generate Cover Letter'):
    input_data = f"Job Title: {job_title}, Preferred Qualifications: {preferred_qualifications}, Hiring Company: {hiring_company}, Applicant Name: {applicant_name}, Past Working Experience: {past_experience}, Current Working Experience: {current_experience}, Skillsets: {skillsets}, Qualifications: {qualifications}"
    cover_letter = generate_cover_letter(input_data)
    st.text_area("Cover Letter:", cover_letter, height=300)

