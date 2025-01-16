# Dockerfile
FROM apache/spark:3.4.1

USER root

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application files
WORKDIR /app
COPY movie_recommender.py .

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_PYTHON=python3

# Expose ports
EXPOSE 4040

# Command to run the application
CMD ["spark-submit", "--master", "local[*]", "movie_recommender.py"]