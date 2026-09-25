# Sepsis Prediction Dashboard

Full-stack dashboard for the PhysioNet 2019 Sepsis Early Warning System.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend       │
│   (React +      │     │   (FastAPI)     │
│    TypeScript)  │     │                 │
│   Port 5173     │     │   Port 8000     │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐  ┌──────────┐  ┌──────────┐
              │ Models  │  │ XAI      │  │ Database │
              │ (.pkl,  │  │ Artifacts│  │ (SQLite) │
              │  .cbm)  │  │ (SHAP/   │  │          │
              └─────────┘  │  LIME)   │  └──────────┘
                           └──────────┘
```

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
cd enhanced/dashboard
docker-compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Local Development

**Backend:**
```bash
cd enhanced/dashboard/backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd enhanced/dashboard/frontend
npm install
npm run dev
```

## Features

| Page | Description |
|------|-------------|
| **Overview** | Model metrics, global/temporal SHAP importance, external validation |
| **Patients** | Browse 40K+ patients, filter by sepsis status, view hourly timelines |
| **Explainability** | Per-patient SHAP waterfall + LIME, global/temporal importance charts |
| **Predict** | Real-time sepsis risk prediction with clinical action guidance |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/metrics/overview` | Complete model metrics + feature importance |
| `GET /api/metrics/models` | Individual model performance |
| `GET /api/metrics/external` | External validation (Set A / Set B) |
| `GET /api/patients` | Paginated patient list |
| `GET /api/patients/{id}/records` | Hourly records for a patient |
| `POST /api/predictions` | Make prediction for new patient |
| `GET /api/explanations/patient/{id}/{iculos}` | SHAP + LIME for patient-hour |
| `GET /api/explanations/global/shap` | Global SHAP importance |
| `GET /api/explanations/temporal/shap` | Temporal SHAP importance |

## Model Artifacts (Auto-mounted)

The backend expects these files (from the main pipeline):

```
enhanced/
├── models/
│   ├── catboost_model.cbm
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── rf_model.pkl
│   ├── meta_learner.pkl
│   ├── calibrator.pkl
│   ├── optimal_threshold.json
│   ├── calibration_info.json
│   └── transformers/
│       ├── scaler_robust.pkl
│       └── imputer_knn.pkl
├── experiments/
│   ├── selected_features.json
│   ├── external_validation_setA_setB.json
│   └── xai/
│       ├── global_shap_beeswarm.png
│       ├── temporal_shap_importance.png
│       ├── patient_waterfall_TP.png
│       ├── patient_waterfall_TN.png
│       ├── patient_lime_TP.png
│       └── patient_lime_TN.png
└── data/processed/
    └── test_temporal.parquet
```

Run the main pipeline first to generate these:
```bash
# From project root
python enhanced/data/preprocessing.py
python enhanced/features/temporal.py
python enhanced/features/selection.py
python enhanced/models/train_xgb.py
python enhanced/models/train_lgbm.py
python enhanced/models/train_catboost.py
python enhanced/models/train_rf.py
python enhanced/stacking/stack.py
python enhanced/calibration/calibrate.py
python enhanced/calibration/threshold.py
python enhanced/xai/explain.py
```

## Database

SQLite database (`enhanced_dashboard.db`) is created automatically on first run with tables:
- `patients` - Patient metadata
- `hourly_records` - Hourly vitals/labs
- `predictions` - Prediction audit trail
- `models` - Model registry

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///enhanced_dashboard.db` | Database connection |

## Screenshots

*(Add screenshots after running)*

## License

Part of the PhysioNet 2019 Sepsis Early Warning project.