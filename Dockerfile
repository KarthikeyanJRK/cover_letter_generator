# Use an official Python runtime as a base image
FROM python:3.11

# Set the working directory to /app inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the local code to the container's workspace
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download necessary NLTK datasets
RUN python -m nltk.downloader stopwords punkt averaged_perceptron_tagger wordnet omw-1.4

# Install SpaCy English model
RUN pip install spacy && \
    python -m spacy download en_core_web_sm

# Install pyresparser and ensure the config file is in place if missing
RUN pip install --force-reinstall pyresparser
COPY config.cfg /usr/local/lib/python3.11/site-packages/pyresparser/ || echo "Config file not found in local directory, skipping copy."

# Expose the port Streamlit will run on
EXPOSE 8501

# Set the default command to execute
# when creating a new container from the image
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

