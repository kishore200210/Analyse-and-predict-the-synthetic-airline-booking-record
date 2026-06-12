# Data Science Task Overview

This project now includes three areas of work for the booking and complaint dataset:

## 1. Data Analysis
Use this folder to explore the booking dataset, inspect patterns, and create visual insights.

## 2. Predictive Modelling
Use this folder to test booking outcome prediction models and compare performance.

## 3. GenAI Bonus Challenge
Use this folder for complaint-text analysis, category classification, and LLM-based experimentation.

## Current Demo and API
Two runnable components are available in the root folder:

- app.py: Streamlit demo for complaint classification and summary view
- route/api.py: FastAPI API with:
  - POST /classify
  - GET /health

## How to run

### Streamlit demo
From the project root:
```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### FastAPI API
From the project root:
```bash
.\.venv\Scripts\python.exe -m uvicorn route.api:app --reload
```

Then open:
- http://127.0.0.1:8000/docs for the API docs
- http://127.0.0.1:8000/health for the health check

## Quick summary
- Task 1: understand the data
- Task 2: predict booking outcomes
- Task 3: analyze complaint text and classify issues

If you want to start in order, open:
1. Data Analysis
2. Predictive Modelling
3. GenAI Bonus Challenge
