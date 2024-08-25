# Machine Learning Task - Revised approach

### Candidate
- Mattia Delleani 

# IMPORTANT: Revised version

In response to the feedback, I have reorganized the project structure and created a new folder named `scratch`, containing the notebook called `assigment_revised.ipynb`.

## Main components

- Open and run the `assignment_revised.ipynb` to see the proposed revised solution.

- Run the pipeline for training and evaluation: 
    ```python 
    python main.py [OPTIONS]
    ```

    check the accepted arguments by running:
    ```python 
    python main.py --help
    ```

**Structure**:

```plaintext
.
├── scratch # folder with ECG data
│   │
│   ├── modules # custom modules implemented from scratch:
│   │   ├── utils.py
│   │   ├── networks.py
│   │   ├── datasets.py
│   │   ├── data_preprocessor.py
│   │   └── visualization.py
│   ├── experiments # with results
│   │ 
│   ├── assignment_revised.ipynb # Jupyter Notebook revised
│   ├── main.py # file for cmd line training
│   └── README.md

```

This revised assignment is based on the feedback provided: so I remove the dependencies on existing solution and provided my own implementation, especially for the data preprocessing, ML model and training. In the end I added new components (i.e GradCAM for model explainability).

## Revised Implementation

In response to the feedback provided, I have reorganized the project structure and created a new folder named `scratch`, which contains this Jupyter Notebook. The revised version incorporates several custom modules developed from scratch:

- `modules.utils`
- `modules.networks`
- `modules.datasets`
- `modules.data_preprocessor`
- `modules.visualization`

These modules were designed to address specific project needs, featuring custom functions and classes for effective data processing, model architecture, and visualization.

## Key Areas of Focus

- **Data Processing**: Custom functions were added to efficiently preprocess and handle ECG data, ensuring that it is well-prepared for subsequent analysis.
- **Model Training, Validation, and Testing**: The entire pipeline for training and evaluating the model was built from scratch, allowing for flexibility and precise control over the training process and evaluation metrics.
- **Grad-CAM Integration**: Grad-CAM was integrated to provide insights into the model’s focus on different segments of the ECG signal, enhancing model interpretability.

