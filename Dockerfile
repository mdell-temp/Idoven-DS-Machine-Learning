# Use Python 3.10 as the base image
FROM python:3.10

# Install wget and ca-certificates
RUN apt-get update && apt-get install -y wget ca-certificates && update-ca-certificates

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the repo to the container
COPY . .

# If you want to download, remove the comment to run the data download script
# RUN chmod +x /app/data/download_data.sh
# RUN /app/data/download_data.sh

# Create and set the working directory for src
RUN mkdir -p /app/src
WORKDIR /app/src

# Install JupyterLab
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install the dependencies from requirements.txt
RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install jupyterlab

# Install PyTorch with CUDA (Make sure to use the correct version)
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Expose the port that JupyterLab will run on
EXPOSE 8888

# Define the command to run when the container starts
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]