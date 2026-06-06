# Road Damage Detection for Smart Cities

An end-to-end road damage classification project built using Deep Learning, FastAPI, PostgreSQL, and PostGIS.

The system can classify road damage from uploaded images, store reports with GPS coordinates, and search for nearby damage reports using geospatial queries.

---

## Project Overview

This project was developed to understand the complete machine learning workflow:

1. Data Preparation
2. Model Training
3. Model Evaluation
4. Model Deployment
5. Database Integration
6. Geospatial Search

The project uses a ResNet50 transfer learning model to classify road damage into different categories and serves predictions through a FastAPI backend.

---

## Features

- Road damage classification using ResNet50
- FastAPI-based REST API
- PostgreSQL database integration
- PostGIS support for geospatial queries
- Image upload and storage
- Nearby damage search using GPS coordinates
- Docker support

---

## Tech Stack

### Machine Learning

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-Learn

### Backend

- FastAPI
- Uvicorn

### Database

- PostgreSQL
- PostGIS
- SQLAlchemy

### Deployment

- Docker
- Docker Compose

---

## Project Structure

```text
road-damage-detection/

├── app/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── schemas/
│   ├── services/
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── uploads/
│
├── ml/
│   ├── models/
│   ├── reports/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── predict.py
│   └── prepare_rdd_classification.py
│
├── notebooks/
│   ├── 03_test_saved_model.ipynb/
│   ├── 04_self_contained_professional_cnn.ipynb/
│   ├── 05_self_contained_resnet50_transfer_learning.ipynb
│
├── sql/
│   └── geospatial_queries.sql
│   └── schema.sql
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Dataset Structure

The dataset should be organized as follows:

```text
data/processed/

├── train/
│   ├── crack/
│   ├── pothole/
│   └── manhole/
│
├── val/
│   ├── crack/
│   ├── pothole/
│   └── manhole/
│
└── test/
    ├── crack/
    ├── pothole/
    └── manhole/
```

Each folder name automatically becomes the class label during training.

---

## Model Training

The project uses transfer learning with ResNet50.

### Training Steps

1. Load dataset
2. Apply data augmentation
3. Load pretrained ResNet50 weights
4. Train classifier head
5. Fine-tune ResNet50 layers
6. Save trained model

### Output Files

```text
ml/models/

road_damage_resnet50.keras
road_damage_resnet50_labels.json
```

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score

Example metrics:

```text
Accuracy : 85%

Class-wise Performance

Crack
Precision : 0.94
Recall    : 0.90

Manhole
Precision : 0.67
Recall    : 0.84

Pothole
Precision : 0.69
Recall    : 0.68
```

---

## FastAPI Endpoints

### Health Check

```http
GET /health
```

Returns:

```json
{
  "status": "ok"
}
```

---

### Predict Damage

```http
POST /predict
```

Upload an image and get a prediction.

Response:

```json
{
  "damage_type": "pothole",
  "severity": "high",
  "confidence": 0.91
}
```

---

### Create Report

```http
POST /reports
```

Required fields:

- image
- latitude
- longitude

This endpoint:

1. Predicts road damage
2. Stores report in PostgreSQL
3. Saves image path
4. Stores GPS location in PostGIS

---

### Get Reports

```http
GET /reports
```

Returns stored damage reports.

Example:

```json
[
  {
    "id": 1,
    "damage_type": "pothole",
    "severity": "high",
    "confidence": 0.91
  }
]
```

---

### Search Nearby Reports

```http
GET /reports/nearby
```

Parameters:

```text
lat
lon
radius_km
```

Example:

```http
GET /reports/nearby?lat=25.5941&lon=85.1376&radius_km=5
```

Returns all reports within a 5 km radius.

---

## Database Schema

The project stores road damage reports in PostgreSQL.

Main fields:

```text
id
damage_type
severity
confidence
latitude
longitude
location
image_path
created_at
```

The `location` field is stored as a PostGIS geography point and is used for nearby searches.

---

## How Nearby Search Works

Each report is stored with GPS coordinates.

Example:

```text
Latitude  = 25.5941
Longitude = 85.1376
```

When a user requests:

```http
GET /reports/nearby
```

PostGIS calculates the distance between the provided coordinates and all stored reports.

Only reports inside the requested radius are returned.

---

## Running the Project

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Configure Database

Create PostgreSQL database:

```sql
CREATE DATABASE road_damage;
```

Run schema:

```sql
\i sql/schema.sql
```

---

### Start FastAPI

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Future Improvements

- Road damage object detection
- Mobile application integration
- Automatic GPS extraction
- Severity estimation model
- Interactive dashboard
- Real-time road monitoring

---

## Learning Outcomes

Through this project, I learned:

- Transfer Learning with ResNet50
- Image Classification
- FastAPI Development
- PostgreSQL Integration
- PostGIS Geospatial Queries
- SQLAlchemy ORM
- Model Deployment
- REST API Design
- Docker Basics

---

## Author

Shubham Kumar Jha
