# Machine Learning Task

### Candidate
-

## Introduction

Welcome to this project! Here, we focus on implementing a machine learning (ML) classification model from scratch using the provided data. The objective is to explore the problem, assess reasonable and interesting approaches within the given time constraints, and make design decisions accordingly. This repository will guide you through the entire process, from reading and analyzing ECG files to visualizing the signals and building a model that can provide actionable insights.

At Idoven, data scientists work with anonymized patient data to:
1. Read ECG files and their corresponding annotations.
2. Process and visualize the signals in a manner that doctors can interpret.
3. Build models to classify and analyze the signals.

## Repository Structure

Here's the structure of the repository:

```plaintext
.
├── data
│   ├── ptbxl
│   ├── download_data.sh
│   └── README.md
├── src
│   ├── assignment.ipynb
│   ├── data
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── data_augmentation.py
│   ├── utils
│   │   ├── utilities.py
│   │   └── data_augmentation.py
│   ├── requirements.txt
│   └── README.md
├── Dockerfile
├── README.md
└── references
    └── reference_documentation.pdf

<!-- ├── scripts
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── evaluate_model.py -->

```
## Getting Started

### Prerequisites

Ensure you have the following installed:
- Git
- Docker

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo.git # clone the repo
cd your-repo # move inside the repo folder
```

**Docker** 

The following instructions are for Windows based user, there might be some differences for other OS.

Before running the Docker, check these:
1. Open the *Dockerfile* and comment ```RUN /app/data/download_data.sh``` if you already have downloaded the data.
2. Open the *src/requirements.txt* and modify the pytorch-cuda installation (i.e. ```RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```) according to your GPU requirements.

Run the Docker:

1. Build the Docker image:

```bash
docker build -t ecg-classification .
```

2. Run the docker container:
```bash
# for Ubuntu replace %cd%\ with $pwd
docker run -p 8888:8888 -v %cd%\src:/app/src ecg-classification # CPU
# or
docker run --gpus=all -p 8888:8888 -v %cd%\src:/app/src ecg-classification # GPU
```
This will start a Docker container and expose port 8888. You can access Jupyter Notebook by navigating to http://localhost:8888 in your web browser.



