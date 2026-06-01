# Road Damage Detection for Smart Cities

End-to-end road damage classification project for a final-year portfolio/resume.

The project starts with a **custom CNN baseline** and then improves it with **ResNet50 transfer learning**. The trained model is served through **FastAPI**, and prediction reports are stored in **PostgreSQL/PostGIS** so severe road damage can be queried by GPS radius.

## Why This Project Is Built This Way

Interview story:

1. Train a basic CNN from scratch to establish a baseline.
2. Train ResNet50 with transfer learning to improve feature extraction and validation score.
3. Compare both models using accuracy, precision, recall, F1-score, confusion matrix, and inference time.
4. Deploy the better model behind a FastAPI service.
5. Store GPS-based reports in PostGIS for smart-city repair prioritization.

## Tech Stack

- Python
- TensorFlow/Keras
- OpenCV
- FastAPI
- PostgreSQL + PostGIS
- SQLAlchemy
- Docker + Docker Compose

## Project Structure

```text
app/                    FastAPI backend
ml/                     Training, preprocessing, evaluation, prediction
notebooks/              Colab/Jupyter workflow notes
sql/                    PostGIS schema and geospatial queries
data/raw/               Original RDD dataset files
data/processed/         Classification-ready dataset folders
ml/models/              Exported .keras models and label files
ml/reports/             Evaluation reports and confusion matrices
```

## Dataset Format

For the training scripts, prepare data like this:

```text
data/processed/
  train/
    longitudinal_crack/
    transverse_crack/
    alligator_crack/
    pothole/
    rutting/
    no_damage/
  val/
    ...
  test/
    ...
```

If you use RDD annotations directly, run the preparation script:

```bash
python -m ml.prepare_rdd_classification \
  --images-dir data/raw/images \
  --annotations-dir data/raw/annotations \
  --output-dir data/processed \
  --mode crop
```

`--mode crop` usually gives stronger classification performance because each training sample focuses on the actual damaged region. Use `--mode image` if you want one label per full dashcam image.

## Train Baseline CNN

```bash
python -m ml.train --model cnn --data-dir data/processed --epochs 25
```

## Train ResNet50

```bash
python -m ml.train --model resnet50 --data-dir data/processed --epochs 20 --fine-tune-epochs 10
```

## Evaluate a Model

```bash
python -m ml.evaluate \
  --model-path ml/models/road_damage_resnet50.keras \
  --labels-path ml/models/road_damage_resnet50_labels.json \
  --data-dir data/processed
```

## Run API Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start only the API:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## Run With Docker

```bash
docker compose up --build
```

The API will run at:

```text
http://localhost:8000/docs
```

PostgreSQL/PostGIS will run on port `5432`.

## Important Endpoints

```text
GET  /health
POST /predict
POST /reports
GET  /reports
GET  /reports/nearby?lat=28.6139&lon=77.2090&radius_km=5&severity=high
```

## Resume Line

Built an end-to-end road damage classification system using TensorFlow, OpenCV, FastAPI, Docker, and PostgreSQL/PostGIS. Compared a custom CNN baseline against ResNet50 transfer learning, served the best model through an API, and implemented geospatial SQL queries to prioritize severe damage reports within a given radius.

