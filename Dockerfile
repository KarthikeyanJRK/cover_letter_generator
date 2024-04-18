FROM python:3.8

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
RUN pip install -r requirements.txt

# Install nltk and specific version of spacy
RUN pip install nltk
RUN pip install spacy==2.3.5

# Install specific version of spacy model directly from URL
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

# Install pyresparser
RUN pip install pyresparser

# Download necessary NLTK datasets
RUN python -m nltk.downloader stopwords punkt averaged_perceptron_tagger wordnet omw-1.4

# Set the default command to execute
# when creating a new container from the image
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
