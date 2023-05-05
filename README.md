# Student Performance Prediction

This project aimed to develop a system for predicting undergraduate students' performance in information technology studies to support admission decisions. The project consisted of two main parts: analysis and model development and web application interface development. The study found that the grade point average of secondary school, science, English language, and mathematics were the most related factors to the grade point average of the information technology curriculum. Five machine learning algorithms were studied.

## Tech Stack
- Analysis and Model Development:
  - Python
  - Scikit-learn
- Web Application Interface Development:
  - FastAPI
  - MongoDB
- Model Algorithms:
  - K-Nearest Neighbor
  - Decision Tree
  - Random Forest
  - Multi-Layer Perceptron
  - Logistic Regression
- Data Analysis:
  - Pearson Correlation Analysis

## How to run with Docker

```bash
# Build Docker Image
docker build -t fastapi-cd:1.0 .

# Run API service on port 8000
docker run -p 8000:8000 --name fastapi fastapi-cd:1.0
```
