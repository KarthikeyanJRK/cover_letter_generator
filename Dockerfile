FROM python:3.11

# Expose the port that Streamlit will run on
EXPOSE 8501

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords punkt averaged_perceptron_tagger wordnet omw-1.4

# Download SpaCy English model
RUN pip3 install spacy && python -m spacy download en_core_web_sm

# Set the default command to execute
# when creating a new container from the image
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

