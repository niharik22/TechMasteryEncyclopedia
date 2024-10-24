# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install dependencies for Chrome and ChromeDriver
RUN apt-get update && apt-get install -y wget gnupg unzip \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update && apt-get install -y google-chrome-stable \
    && wget -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip \
    && unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/ \
    && rm /tmp/chromedriver.zip \
    && apt-get clean

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

## Declare a volume to persist data
#VOLUME ["/app/data"]

# Ensure the script is executable
RUN chmod +x /app/run.sh

# Set environment variables for ChromeDriver and display settings
ENV DISPLAY=:99

# Command to run the shell script
CMD ["/app/run_all.sh"]