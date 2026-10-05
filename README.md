# Mental Health Score Predictor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)

A full-stack machine learning project that predicts a student's **mental health score (0-10)** from their social media usage, sleep, study, physical activity, and stress level. It includes the model training notebook, a FastAPI backend, and a web frontend.

- **Live demo:** [add your frontend link here]
- **API:** https://mental-health-score-analyser.onrender.com
- **Interactive API docs:** https://mental-health-score-analyser.onrender.com/docs

> **Note:** The API runs on Render. If it has been idle, the first request may take a short while to wake up.

<!-- Add a screenshot of the app here: ![App screenshot](assets/screenshot.png) -->

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## Overview

The app takes 12 inputs about a student's daily habits and returns a predicted mental health score, shown on a gauge in the browser.

1. The user fills in the form on the frontend.
2. JavaScript validates the inputs and sends them to `POST /predict`.
3. FastAPI validates the request with Pydantic and groups the country (the top countries from training are kept, everything else becomes `Other`).
4. The trained scikit-learn pipeline preprocesses the data and predicts the score.
5. The frontend displays the score with a short interpretation.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data analysis and modeling | Python, pandas, NumPy, seaborn, matplotlib, scikit-learn |
| Backend | FastAPI, Pydantic, Uvicorn, joblib |
| Frontend | HTML, CSS, JavaScript (no framework) |
| Hosting | Render (API) |

## Dataset

**Student Social Media and Mental Health Impact**: 5,000 records, 13 columns.

| Type | Columns |
|------|---------|
| Numeric | `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Study_Hours`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` |
| Categorical | `Gender`, `Country`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use` |
| Ordinal | `Stress_Level` (Low < Medium < High < Very High) |
| Target | `Mental_Health_Score` |

The raw file is included in this repo: `Student_Social_Media_And_Mental_Health_Impact.csv`. [add original source link]

**Data cleaning**
- No missing values; 2 duplicate rows removed
- Negative `Physical_Activity_Hours` values clipped to 0
- `Country` grouped into the top 10 values plus `Other` to limit the number of one-hot columns

## Model

All preprocessing and the model live in a single scikit-learn `Pipeline`, and the train/test split (70/30) happens **before** any fitting to avoid data leakage.

| Feature group | Preprocessing |
|---------------|---------------|
| `Study_Hours` (slightly skewed) | `log1p` transform, then `StandardScaler` |
| Other numeric columns | `StandardScaler` |
| `Stress_Level` | `OrdinalEncoder` (Low, Medium, High, Very High) |
| Gender, academic level, platform, purpose, grouped country | `OneHotEncoder(handle_unknown="ignore")` |

**Results on the held-out test set**

| Model | Test R² | Train R² | MAE |
|-------|---------|----------|-----|
| Linear Regression (baseline) | 0.740 | 0.724 | 0.536 |
| **Random Forest (default), deployed** | **0.878** | 0.981 | **0.347** |
| Random Forest (tuned, `RandomizedSearchCV`) | 0.865 | 0.955 | 0.369 |

The default Random Forest scored best on the test set and is the model saved as `Mental_Health_Model.pkl`. The gap between its train and test R² shows some overfitting, though the tuned model (which narrows the gap) did not improve test performance.

## Project Structure

```
.
├── main.py                                            # FastAPI application
├── Mental_Health_Model.pkl                            # Trained pipeline (preprocessing + model)
├── ML_Python.ipynb                                    # EDA, feature engineering, training
├── Student_Social_Media_And_Mental_Health_Impact.csv  # Dataset
├── requirements.txt                                   # Python dependencies
├── index.html                                         # Frontend page
├── style.css                                          # Frontend styles
├── script.js                                          # Frontend logic and API calls
└── README.md
```

## API Reference

### `GET /`

Health check. Returns a welcome message.

### `POST /predict`

Returns the predicted mental health score.

**Request body**

```json
{
  "age": 20,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Networking",
  "avg_daily_usage_hours": 5,
  "daily_unlocks": 50,
  "study_hours": 4,
  "physical_activity_hours": 1,
  "sleep_hours_per_night": 7,
  "stress_level": "Medium"
}
```

**Response**

```json
{
  "predicted_mental_health_score": 6.5
}
```

**Field rules**

| Field | Type | Allowed values |
|-------|------|----------------|
| `age` | integer | 10 to 100 |
| `gender` | string | `Male`, `Female` |
| `country` | string | Any text (unknown countries are treated as `Other`) |
| `academic_level` | string | `High School`, `Undergraduate`, `Graduate` |
| `most_used_platform` | string | `Facebook`, `LinkedIn`, `Instagram`, `Snapchat`, `Twitter`, `YouTube`, `TikTok`, `LINE`, `KakaoTalk`, `VKontakte`, `WhatsApp`, `WeChat` |
| `purpose_of_use` | string | `Networking`, `Education`, `Entertainment`, `News` |
| `avg_daily_usage_hours` | float | 0 to 24 |
| `daily_unlocks` | integer | 0 or more |
| `study_hours` | float | 0 to 24 |
| `physical_activity_hours` | float | 0 to 24 |
| `sleep_hours_per_night` | float | 0 to 24 |
| `stress_level` | string | `Low`, `Medium`, `High`, `Very High` |

Invalid input returns a `422` response describing the failing fields.

**Example with cURL**

```bash
curl -X POST 'https://mental-health-score-analyser.onrender.com/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 20,
    "gender": "Male",
    "country": "India",
    "academic_level": "Undergraduate",
    "most_used_platform": "Instagram",
    "purpose_of_use": "Networking",
    "avg_daily_usage_hours": 5,
    "daily_unlocks": 50,
    "study_hours": 4,
    "physical_activity_hours": 1,
    "sleep_hours_per_night": 7,
    "stress_level": "Medium"
  }'
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- pip

### 1. Clone the repository

```bash
git clone https://github.com/[your-username]/[your-repo].git
cd [your-repo]
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn main:app --reload
```

The API is now available at `http://127.0.0.1:8000`, with docs at `http://127.0.0.1:8000/docs`.

### 4. Run the frontend

In `script.js`, point `API_BASE` to your local server:

```javascript
const API_BASE = "http://127.0.0.1:8000";
```

Then open `index.html` in your browser.

> **Important:** `Mental_Health_Model.pkl` was saved with a specific scikit-learn version. Keep the version in `requirements.txt` the same as the one used for training, or loading the model may fail or give wrong results.

### Retraining the model

Open `ML_Python.ipynb`, update the dataset path in the loading cell, and run all cells. The last cell saves a new `Mental_Health_Model.pkl`.

## Limitations

- **Narrow training range.** The training data covers ages 18-24, about 1-9 hours of daily usage, and 0.3-8.3 study hours. The API accepts wider values, but predictions outside the training range are less reliable.
- **Dataset scope.** The model learns patterns from one dataset of 5,000 students and may not generalize to other populations or countries.
- **Overfitting.** The Random Forest fits the training data much more closely than the test data.
- **Correlation, not causation.** The model finds statistical patterns. It does not show that any habit causes a change in mental health.

## Future Improvements

- Add cross-validation and feature-importance analysis to the notebook
- Add automated tests for the API
- Show the main factors behind each prediction
- Add a Dockerfile for easier deployment

## Disclaimer

This project is for **educational and informational purposes only**. It is not a clinical tool and must not be used for diagnosis or treatment decisions. If you are struggling with your mental health, please talk to a qualified professional or someone you trust.

## Author

**[Your Name]**
GitHub: [@your-username](https://github.com/your-username) · LinkedIn: [your-profile](https://linkedin.com/in/your-profile)
