# Particle Coordinate Prediction in RSD Sensors
### Multi-Output Regression Task — Politecnico di Torino

## Overview
The aim of this project was to build a multi-output regression pipeline 
to predict the coordinates (x, y) indicating where a particle of interest 
passed, based on data collected from signals in an RSD sensor 
(Resistive Silicon Detector).

A variety of preprocessing techniques, dimensionality reduction methods, 
and ML models were tested. A Random Forest Regressor achieved the best 
score, outperforming the competition baseline.

**Best result: Euclidean Distance = 5.417** (baseline: 6.629)

## Project Structure
```
Particles_RSD_Sensors_RegressionTask/
├── data/
│   ├── raw/                  ← development.csv, evaluation.csv
│   └── processed/            ← cleaned data (optional)
├── models/                   ← saved trained model (.pkl)
├── notebooks/                ← EDA, experiments, model comparison
├── outputs/                  ← predictions CSV and hexbin plot
├── reports/                  ← full project report (PDF)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← loading and cleaning
│   ├── preprocessor.py       ← feature selection and scaling
│   ├── trainer.py            ← model training and evaluation
│   └── predictor.py          ← inference and output generation
├── main.py                   ← full pipeline entry point
└── requirements.txt
```

## Workflow
Base experiments, EDA, and model comparisons are documented in 
`notebooks/Report_ParticlesinRSDSensors.ipynb`. This includes all 
tested configurations: linear models (Linear Regression, Lasso, Ridge), 
PCA experiments, Decision Tree, and Random Forest variants.

The final best model pipeline is implemented as a clean, reproducible 
Python package in `src/` and runs end-to-end via `main.py`.

## How to Run

```bash
# 1. Clone the repo
git clone _____

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python main.py
```

Output files will be saved to `outputs/`.

## Results

| Model | Features | Euclidean Distance |
|---|---|---|
| Linear Regression | 90 | 17.352 |
| Linear Regression | 60 | 18.157 |
| Linear + PCA (30 components) | 30 | 176.953 |
| Decision Tree | 90 | 8.370 |
| Random Forest | 90 | 5.585 |
| **Random Forest (top 38 features, 150 estimators)** | **38** | **5.417** |

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Authors
Silva Bashllari & Baharak Qaderi — Politecnico di Torino

## License
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
