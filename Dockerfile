FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt --no-cache-dir

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "code/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
