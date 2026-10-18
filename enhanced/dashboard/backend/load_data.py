"""Script to load PhysioNet 2019 data into SQLite database for dashboard."""
import pandas as pd
import json
from pathlib import Path
from sqlalchemy.orm import Session
from app.database import engine, SessionLocal, init_db
from app.models import Patient, HourlyRecord, Base

DATA_DIR = Path(__file__).parent.parent.parent.parent / "enhanced" / "data" / "processed"

def load_patients_from_temporal(temporal_path: Path, limit: int = 1000):
    """Load patient metadata and hourly records from temporal parquet."""
    print(f"Loading data from {temporal_path}...")
    df = pd.read_parquet(temporal_path)

    # Get unique patients (limit for performance)
    patient_ids = df['patient_id'].unique()[:limit]
    print(f"Found {len(patient_ids)} patients (limited to {limit})")

    db = SessionLocal()
    try:
        for pid in patient_ids:
            patient_df = df[df['patient_id'] == pid].sort_values('ICULOS')

            # Patient metadata from first row
            first_row = patient_df.iloc[0]

            # Check if patient already exists
            existing = db.query(Patient).filter(Patient.patient_id == pid).first()
            if existing:
                continue

            patient = Patient(
                patient_id=pid,
                age=float(first_row.get('Age', 0)) if pd.notna(first_row.get('Age')) else None,
                gender=int(first_row.get('Gender', 0)) if pd.notna(first_row.get('Gender')) else None,
                unit1=int(first_row.get('Unit1', 0)) if pd.notna(first_row.get('Unit1')) else None,
                unit2=int(first_row.get('Unit2', 0)) if pd.notna(first_row.get('Unit2')) else None,
                hosp_adm_time=float(first_row.get('HospAdmTime', 0)) if pd.notna(first_row.get('HospAdmTime')) else None,
                sepsis_label=int(first_row.get('SepsisLabel', 0)),
                onset_hour=int(first_row.get('onset_hour', 0)) if pd.notna(first_row.get('onset_hour')) and first_row.get('onset_hour') > 0 else None,
            )
            db.add(patient)

            # Add hourly records (limit to first 48 hours for performance)
            for _, row in patient_df.head(48).iterrows():
                record = HourlyRecord(
                    patient_id=pid,
                    iculos=int(row.get('ICULOS', 0)),
                    hr=float(row['HR']) if pd.notna(row.get('HR')) else None,
                    o2sat=float(row['O2Sat']) if pd.notna(row.get('O2Sat')) else None,
                    temp=float(row['Temp']) if pd.notna(row.get('Temp')) else None,
                    sbp=float(row['SBP']) if pd.notna(row.get('SBP')) else None,
                    map=float(row['MAP']) if pd.notna(row.get('MAP')) else None,
                    dbp=float(row['DBP']) if pd.notna(row.get('DBP')) else None,
                    resp=float(row['Resp']) if pd.notna(row.get('Resp')) else None,
                    etco2=float(row['etco2']) if pd.notna(row.get('etco2')) else None,
                    baseexcess=float(row['baseexcess']) if pd.notna(row.get('baseexcess')) else None,
                    hco3=float(row['hco3']) if pd.notna(row.get('hco3')) else None,
                    fio2=float(row['FiO2']) if pd.notna(row.get('FiO2')) else None,
                    paco2=float(row['PaCO2']) if pd.notna(row.get('PaCO2')) else None,
                    sao2=float(row['SaO2']) if pd.notna(row.get('SaO2')) else None,
                    ast=float(row['ast']) if pd.notna(row.get('ast')) else None,
                    bun=float(row['BUN']) if pd.notna(row.get('BUN')) else None,
                    alkphos=float(row['alkphos']) if pd.notna(row.get('alkphos')) else None,
                    calcium=float(row['Calcium']) if pd.notna(row.get('Calcium')) else None,
                    chloride=float(row['chloride']) if pd.notna(row.get('chloride')) else None,
                    creatinine=float(row['creatinine']) if pd.notna(row.get('creatinine')) else None,
                    bilirubin_direct=float(row['bilirubin_direct']) if pd.notna(row.get('bilirubin_direct')) else None,
                    glucose=float(row['Glucose']) if pd.notna(row.get('Glucose')) else None,
                    lactate=float(row['lactate']) if pd.notna(row.get('lactate')) else None,
                    magnesium=float(row['magnesium']) if pd.notna(row.get('magnesium')) else None,
                    phosphate=float(row['phosphate']) if pd.notna(row.get('phosphate')) else None,
                    potassium=float(row['Potassium']) if pd.notna(row.get('Potassium')) else None,
                    bilirubin_total=float(row['bilirubin_total']) if pd.notna(row.get('bilirubin_total')) else None,
                    troponini=float(row['troponini']) if pd.notna(row.get('troponini')) else None,
                    hct=float(row['Hct']) if pd.notna(row.get('Hct')) else None,
                    hgb=float(row['Hgb']) if pd.notna(row.get('Hgb')) else None,
                    ptt=float(row['ptt']) if pd.notna(row.get('ptt')) else None,
                    wbc=float(row['WBC']) if pd.notna(row.get('WBC')) else None,
                    fibrinogen=float(row['fibrinogen']) if pd.notna(row.get('fibrinogen')) else None,
                    platelets=float(row['Platelets']) if pd.notna(row.get('Platelets')) else None,
                    sepsis_label=int(row.get('SepsisLabel', 0)),
                )
                db.add(record)

            if len(patient_ids) > 100 and patient_ids.tolist().index(pid) % 100 == 0:
                db.commit()
                print(f"  Committed {patient_ids.tolist().index(pid)} patients...")

        db.commit()
        print(f"Successfully loaded {len(patient_ids)} patients")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_db()

    # Load train temporal data (has more patients)
    train_temporal = DATA_DIR / "train_temporal.parquet"
    if train_temporal.exists():
        load_patients_from_temporal(train_temporal, limit=500)

    # Also load some validation/test patients
    val_temporal = DATA_DIR / "val_temporal.parquet"
    if val_temporal.exists():
        load_patients_from_temporal(val_temporal, limit=100)

    test_temporal = DATA_DIR / "test_temporal.parquet"
    if test_temporal.exists():
        load_patients_from_temporal(test_temporal, limit=100)

    print("Data loading complete!")