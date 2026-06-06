# Road Damage Detection for Smart Cities

An end-to-end road damage classification system built with Deep Learning, FastAPI, PostgreSQL, and PostGIS. The system classifies road damage from uploaded images, stores geo-tagged reports with GPS coordinates, and supports radius-based damage search using geospatial queries.

---

## Project Highlights

- **CNN vs ResNet50 comparison** — custom baseline vs. transfer learning with quantified improvement
- **Class imbalance handling** — addressed via `sklearn` balanced class weights applied during training
- **Two-stage fine-tuning** — frozen head training followed by selective unfreezing of top ResNet50 layers
- **Production-ready deployment** — FastAPI + PostgreSQL + PostGIS + Docker
- **Geospatial search** — radius-based nearby damage lookup via PostGIS geography queries

---

## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | TensorFlow / Keras, ResNet50, OpenCV, Scikit-Learn |
| Backend | FastAPI, Uvicorn |
| Database | PostgreSQL, PostGIS, SQLAlchemy |
| Deployment | Docker, Docker Compose |

---

## Dataset

**Source:** [Road Damage Dataset — Potholes, Cracks and Manholes (Kaggle)](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)

The dataset contains real-world road damage images across three classes. It has a significant class imbalance — crack images dominate the training set, while manhole and pothole samples are substantially underrepresented.

| Class | Train | Val | Test |
|---|---|---|---|
| Crack | ~1,570 | ~340 | ~350 |
| Pothole | ~440 | ~80 | ~95 |
| Manhole | ~305 | ~65 | ~65 |

This imbalance directly affects model performance on minority classes and was explicitly handled during training (see Class Imbalance section below).

---

## Model Development

### Approach

Two models were trained and compared:

1. **Custom CNN** — a baseline model built from scratch
2. **ResNet50 Transfer Learning** — pretrained ImageNet weights with a custom classification head and selective fine-tuning

### Class Imbalance Handling

The crack class has ~3–5x more samples than manhole and pothole. To prevent the model from being biased toward the majority class, balanced class weights were computed using `sklearn` and passed to the training loop:

```python
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(class_names),
    y=train_labels,
)
class_weights = {index: float(weight) for index, weight in enumerate(weights)}
```

These weights were applied via `model.fit(..., class_weight=class_weights)`, penalising misclassification of minority classes more heavily.

### ResNet50 Training Strategy

Training was done in two stages:

**Stage 1 — Frozen Base (Head Training)**
- ResNet50 base frozen; only the custom classification head trained
- Head: `GlobalAveragePooling2D → BatchNormalization → Dense(256, relu, L2) → Dropout(0.40) → Softmax`
- Optimizer: Adam (lr=1e-3), Label Smoothing (0.05)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Training time: ~29.6 minutes

**Stage 2 — Fine-Tuning**
- Top 40 ResNet50 layers unfrozen; lower layers kept frozen to preserve general features
- Optimizer: Adam (lr=1e-5) — much lower rate to avoid destroying pretrained weights
- Training time: ~19.3 minutes

### Data Augmentation

```python
tf.keras.layers.RandomFlip("horizontal")
tf.keras.layers.RandomRotation(0.04)
tf.keras.layers.RandomZoom(0.08)
tf.keras.layers.RandomContrast(0.15)
```

---

## Model Comparison

| Metric | Custom CNN | ResNet50 |
|---|---|---|
| Accuracy | 51.57% | **84.90%** |
| Precision (Macro) | 42.49% | **76.74%** |
| Recall (Macro) | 45.03% | **80.59%** |
| F1 Score (Macro) | 40.63% | **78.34%** |
| F1 Score (Weighted) | 54.65% | **85.16%** |
| ROC-AUC (OvR Macro) | 0.6563 | **0.9466** |

ResNet50 outperforms the custom CNN across every metric. The ROC-AUC jump from 0.66 to 0.95 reflects significantly better class separation, not just accuracy gain. The custom CNN's 51.57% accuracy is only marginally above random chance for a 3-class problem, confirming that a shallow architecture is insufficient for this task without pretrained features.

---

## Severity Estimation

Severity is derived from the predicted damage type and model confidence using rule-based thresholds:

```python
def estimate_severity(damage_type: str, confidence: float) -> str:
    if damage_type == "no_damage":
        return "none"
    if damage_type == "pothole" and confidence >= 0.70:
        return "high"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"
```

Potholes are treated as structurally higher-risk than cracks at equivalent confidence levels, reflecting real-world road safety priorities.

---

## Project Structure

```
road-damage-detection/
│
├── app/
│   ├── api/
│   │   ├── health.py
│   │   ├── predict.py
│   │   └── reports.py
│   ├── core/
│   │   └── config.py
│   ├── db/
│   │   └── session.py
│   ├── models/
│   │   └── report.py
│   ├── schemas/
│   │   └── report.py
│   ├── services/
│   │   ├── model_service.py
│   │   └── report_service.py
│   └── main.py
│
├── data/
│   ├── processed/
│   │   ├── train/  (crack / manhole / pothole)
│   │   ├── val/    (crack / manhole / pothole)
│   │   └── test/   (crack / manhole / pothole)
│   ├── raw/
│   ├── samples/
│   └── uploads/
│
├── ml/
│   ├── models/
│   │   ├── road_damage_resnet50.keras
│   │   ├── road_damage_resnet50_labels.json
│   │   ├── road_damage_self_contained_cnn.keras
│   │   └── road_damage_self_contained_cnn_labels.json
│   ├── reports/
│   │   ├── road_damage_resnet50_metrics.json
│   │   ├── road_damage_resnet50_history.csv
│   │   ├── road_damage_self_contained_cnn_metrics.json
│   │   └── road_damage_self_contained_cnn_history.csv
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── models.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── severity.py
│   └── train.py
│
├── notebooks/
│   ├── 03_test_saved_model.ipynb
│   ├── 04_self_contained_cnn.ipynb
│   └── 05_self_contained_resnet50_transfer_learning.ipynb
│
├── sql/
│   ├── schema.sql
│   └── geospatial_queries.sql
│
├── tests/
│   └── test_severity.py
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## API Endpoints

### Health Check
```http
GET /health
```

### Predict Damage
```http
POST /predict
```
Upload an image and receive a damage prediction.

**Response:**
```json
{
  "damage_type": "pothole",
  "severity": "high",
  "confidence": 0.91
}
```

### Create Report
```http
POST /reports
```
Accepts image + GPS coordinates. Runs prediction, stores result in PostgreSQL with PostGIS geography point.

### Get All Reports
```http
GET /reports
```

### Search Nearby Reports
```http
GET /reports/nearby?lat=25.5941&lon=85.1376&radius_km=5
```
Returns all damage reports within a specified radius using PostGIS `ST_DWithin`.

---

## Database Schema

Reports are stored in PostgreSQL with a PostGIS `geography` column for spatial queries.

| Field | Type | Description |
|---|---|---|
| id | Integer | Primary key |
| damage_type | String | crack / pothole / manhole |
| severity | String | low / medium / high |
| confidence | Float | Model prediction confidence |
| latitude | Float | GPS latitude |
| longitude | Float | GPS longitude |
| location | Geography(Point) | PostGIS point for spatial queries |
| image_path | String | Stored image path |
| created_at | DateTime | Report timestamp |

---

## Running the Project

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd road-damage-detection
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 3. Set up database

```sql
CREATE DATABASE road_damage;
\i sql/schema.sql
```

### 4. Run with Docker (recommended)

```bash
docker-compose up --build
```

### 5. Run locally

```bash
uvicorn app.main:app --reload
```

API docs available at: `http://127.0.0.1:8000/docs`

---

## Author

**Shubham Kumar Jha**  
Final Year B.Tech — Computer Engineering  
[GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/)