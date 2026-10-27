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

def load_patients_from_psv(psv_dirs: list, limit: int = 300):
    """Load patient metadata and hourly records directly from raw PSV files."""
    print(f"Loading data from raw PSV files...")
    psv_files = []
    for d in psv_dirs:
        if Path(d).exists():
            psv_files.extend(list(Path(d).glob("*.psv")))

    if not psv_files:
        print("No PSV files found.")
        return

    db = SessionLocal()
    try:
        loaded = 0
        sepsis_count = 0
        # Sort to find both sepsis and non-sepsis patients
        for psv_path in psv_files:
            if loaded >= limit:
                break

            pid = psv_path.stem
            existing = db.query(Patient).filter(Patient.patient_id == pid).first()
            if existing:
                continue

            try:
                patient_df = pd.read_csv(psv_path, sep="|")
            except Exception:
                continue

            if patient_df.empty:
                continue

            col_map = {c.lower(): c for c in patient_df.columns}
            get_val = lambda row, col: row[col_map[col.lower()]] if col.lower() in col_map and pd.notna(row[col_map[col.lower()]]) else None

            first_row = patient_df.iloc[0]
            max_sepsis = int(patient_df['SepsisLabel'].max()) if 'SepsisLabel' in patient_df.columns else 0

            # Ensure we collect some sepsis positive patients
            if max_sepsis == 0 and loaded > limit * 0.8 and sepsis_count < 20:
                continue

            onset_hour = None
            if max_sepsis == 1 and 'SepsisLabel' in patient_df.columns and 'ICULOS' in patient_df.columns:
                pos_hours = patient_df[patient_df['SepsisLabel'] == 1]['ICULOS']
                if not pos_hours.empty:
                    onset_hour = int(pos_hours.min())

            patient = Patient(
                patient_id=pid,
                age=float(get_val(first_row, 'Age')) if get_val(first_row, 'Age') is not None else None,
                gender=int(get_val(first_row, 'Gender')) if get_val(first_row, 'Gender') is not None else None,
                unit1=int(get_val(first_row, 'Unit1')) if get_val(first_row, 'Unit1') is not None else None,
                unit2=int(get_val(first_row, 'Unit2')) if get_val(first_row, 'Unit2') is not None else None,
                hosp_adm_time=float(get_val(first_row, 'HospAdmTime')) if get_val(first_row, 'HospAdmTime') is not None else None,
                sepsis_label=max_sepsis,
                onset_hour=onset_hour,
            )
            db.add(patient)

            for _, row in patient_df.head(48).iterrows():
                f_val = lambda k: float(get_val(row, k)) if get_val(row, k) is not None else None
                record = HourlyRecord(
                    patient_id=pid,
                    iculos=int(get_val(row, 'ICULOS') or 0),
                    hr=f_val('HR'),
                    o2sat=f_val('O2Sat'),
                    temp=f_val('Temp'),
                    sbp=f_val('SBP'),
                    map=f_val('MAP'),
                    dbp=f_val('DBP'),
                    resp=f_val('Resp'),
                    etco2=f_val('EtCO2'),
                    baseexcess=f_val('BaseExcess'),
                    hco3=f_val('HCO3'),
                    fio2=f_val('FiO2'),
                    paco2=f_val('PaCO2'),
                    sao2=f_val('SaO2'),
                    ast=f_val('AST'),
                    bun=f_val('BUN'),
                    alkphos=f_val('Alkalinephos') if get_val(row, 'Alkalinephos') is not None else f_val('alkphos'),
                    calcium=f_val('Calcium'),
                    chloride=f_val('Chloride'),
                    creatinine=f_val('Creatinine'),
                    bilirubin_direct=f_val('Bilirubin_direct'),
                    glucose=f_val('Glucose'),
                    lactate=f_val('Lactate'),
                    magnesium=f_val('Magnesium'),
                    phosphate=f_val('Phosphate'),
                    potassium=f_val('Potassium'),
                    bilirubin_total=f_val('Bilirubin_total'),
                    troponini=f_val('TroponinI'),
                    hct=f_val('Hct'),
                    hgb=f_val('Hgb'),
                    ptt=f_val('PTT'),
                    wbc=f_val('WBC'),
                    fibrinogen=f_val('Fibrinogen'),
                    platelets=f_val('Platelets'),
                    sepsis_label=int(get_val(row, 'SepsisLabel') or 0),
                )
                db.add(record)

            loaded += 1
            if max_sepsis == 1:
                sepsis_count += 1

            if loaded % 50 == 0:
                db.commit()
                print(f"  Committed {loaded} patients (sepsis positive: {sepsis_count})...")

        db.commit()
        print(f"Successfully loaded {loaded} patients into database (sepsis positive: {sepsis_count}).")
    except Exception as e:
        db.rollback()
        print(f"Error loading PSV files: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_db()

    data_loaded = False
    # Load train temporal data (has more patients)
    train_temporal = DATA_DIR / "train_temporal.parquet"
    if train_temporal.exists():
        load_patients_from_temporal(train_temporal, limit=500)
        data_loaded = True

    # Also load some validation/test patients
    val_temporal = DATA_DIR / "val_temporal.parquet"
    if val_temporal.exists():
        load_patients_from_temporal(val_temporal, limit=100)
        data_loaded = True

    test_temporal = DATA_DIR / "test_temporal.parquet"
    if test_temporal.exists():
        load_patients_from_temporal(test_temporal, limit=100)
        data_loaded = True

    if not data_loaded:
        raw_root = Path(__file__).parent.parent.parent.parent / "dataset" / "physionet_sepsis" / "training"
        psv_dirs = [raw_root / "training_setA", raw_root / "training_setB"]
        load_patients_from_psv(psv_dirs, limit=300)

    print("Data loading complete!")