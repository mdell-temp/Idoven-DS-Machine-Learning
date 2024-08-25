# Machine Learning Task - Revised approach

### Candidate
- Mattia Delleani 

# IMPORTANT: Revised version

In response to the feedback, I have reorganized the project structure and created a new folder named `scratch`, containing the notebook called `assigment_revised.ipynb`.

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

This revised assignment showcases a thorough approach to working with anonymized ECG data for machine learning applications in healthcare. The project included comprehensive steps from data loading and preprocessing to model training, validation, and testing, culminating in detailed results visualization.

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

