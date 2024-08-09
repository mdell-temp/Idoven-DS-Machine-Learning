# Machine Learning Task - Proposed approach

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
├── data # folder with ECG data
│   ├── download_data.sh
│   └── README.md
├── src # folder with proposed approach
│   │
│   ├── data # folder with dataloading and processing modules
│   │   ├── data_augmentation.py
│   │   ├── data_plots.py
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── signal_data_processing.py
│   │
│   ├── evaluation # folder with evaluation modules
│   │   └── visualize.py
│   │
│   ├── experiments # folder with experiments 
│   │   ├── EDA/
│   │   ├── logs/
│   │   └── results/
│   │
│   ├── models # folder with ML models 
│   │   ├── architectures.py
│   │   └── pipeline.py
│   │
│   ├── utils # folder with useful functions
│   │   ├── logger.py
│   │   └── utilities.py
│   │
│   ├── assignment.ipynb # Jupyter Notebook as asked
│   ├── main.py # file for cmd line training
│   ├── requirements.txt
│   └── README.md
│   
├── Dockerfile # Consistent setup across different platforms.
└── README.md
```

**Structure Rationale**

The structure is designed for simplicity and clarity, aligning with the assignment requirements. It provides a logical flow from data processing through model evaluation, making it easy to follow and reproduce the steps taken. By organizing code into distinct modules and separating experiments, the structure enhances readability and maintainability, ensuring that the focus remains on answering the assignment questions effectively.

Main component:
- `assignment.ipynb`: The Jupyter Notebook containing the answers to the assignment, structured into sections for EDA, ML classification, and a conclusion with references, providing a comprehensive response to the task.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Git
- Docker

### Installation

Clone the repository:

```bash
git clone https://github.com/mdell-temp/Idoven-DS-Machine-Learning.git # clone the repo
cd Idoven-DS-Machine-Learning # move inside the repo folder
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
    This will start a Docker container and expose port 8888. You can access Jupyter Lab session by navigating to http://localhost:8888 in your web browser.


3. Navigate through the project:

    - Open and run the `assignment.ipynb` to see the proposed solution and perform EDA, train ML models and evaluate the trained models.

    - Run the pipeline from command line: 
        ```python 
        python main.py [OPTIONS]
        ```

        check the accepted arguments by running:
        ```python 
        python main.py --help
        ```

## TODO List

- **Generalize the Pipeline class** for more flexibility and reuse.
- **Incorporate additional model architectures** to explore a wider range of approaches.
- **Implement MLflow, Neptune, or W&B** for interactive experiment tracking.
- **Add test (pyunit, pytest) for continuous integration** to ensure code quality and reliability.
- **Develop a comprehensive benchmarking setup** for comparing multiple models effectively.
- **Document functions in the code** with descriptions of their libraries, arguments, and return values.


## References
- [PhysioNet PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.2/) for foundational knowledge and dataset details.
- [GitHub Repository by Roios](https://github.com/Roios/ptb_ecg_classification/tree/main) for implementation insights and practical approaches.
- [Automated ECG Interpretation Repository](https://github.com/AutoECG/Automated-ECG-Interpretation) for advanced modeling techniques and benchmarks.
- [From ECG signals to images: a transformation based approach for deep learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7959637/) intresting paper.
- [ECG Arrhythmia classification](https://github.com/lxdv/ecg-classification) useful repo.

## License

This project is licensed under the [MIT](https://github.com/mdell-temp/Idoven-DS-Machine-Learning/blob/main/LICENSE) License (following references)

## Notes

`Dockerfile` is kept outside the project folder (*src*) for a production oriented-view where data, models, UI and other components interact. 
