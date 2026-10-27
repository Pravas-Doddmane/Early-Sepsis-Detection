import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Edit3,
  Eye,
  FileText,
  Heart,
  HelpCircle,
  Info,
  Loader2,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import { explanationsApi, PredictionRequest, PredictionResponse, predictionsApi } from '../api';
import RiskBand from '../components/RiskBand';
import ShapBars from '../components/ShapBars';

const vitals = [
  { key: 'HR', label: 'Heart rate', unit: 'bpm', min: 30, max: 200, step: 1 },
  { key: 'O2Sat', label: 'Oxygen saturation', unit: '%', min: 50, max: 100, step: 1 },
  { key: 'Temp', label: 'Temperature', unit: '°C', min: 30, max: 42, step: 0.1 },
  { key: 'SBP', label: 'Systolic pressure', unit: 'mmHg', min: 50, max: 250, step: 1 },
  { key: 'MAP', label: 'Mean pressure', unit: 'mmHg', min: 30, max: 150, step: 1 },
  { key: 'DBP', label: 'Diastolic pressure', unit: 'mmHg', min: 20, max: 150, step: 1 },
  { key: 'Resp', label: 'Respiratory rate', unit: '/min', min: 5, max: 50, step: 1 },
];

const labs = [
  { key: 'FiO2', label: 'FiO2', unit: 'fraction', min: 0.2, max: 1, step: 0.01 },
  { key: 'pH', label: 'pH', unit: '', min: 6.5, max: 8, step: 0.01 },
  { key: 'PaCO2', label: 'PaCO2', unit: 'mmHg', min: 5, max: 150, step: 1 },
  { key: 'SaO2', label: 'SaO2', unit: '%', min: 50, max: 100, step: 1 },
  { key: 'BUN', label: 'BUN', unit: 'mg/dL', min: 0, max: 200, step: 0.1 },
  { key: 'Calcium', label: 'Calcium', unit: 'mg/dL', min: 0, max: 20, step: 0.1 },
  { key: 'Glucose', label: 'Glucose', unit: 'mg/dL', min: 0, max: 1000, step: 1 },
  { key: 'Potassium', label: 'Potassium', unit: 'mmol/L', min: 0, max: 12, step: 0.1 },
  { key: 'Hct', label: 'Hematocrit', unit: '%', min: 0, max: 100, step: 0.1 },
  { key: 'Hgb', label: 'Hemoglobin', unit: 'g/dL', min: 0, max: 30, step: 0.1 },
  { key: 'WBC', label: 'White blood cells', unit: 'K/µL', min: 0, max: 200, step: 0.1 },
  { key: 'Platelets', label: 'Platelets', unit: 'K/µL', min: 0, max: 1500, step: 1 },
];

export const HEALTHY_ADULT_SAMPLE: Record<string, number> = {
  HR: 72,
  O2Sat: 98,
  Temp: 36.8,
  SBP: 120,
  MAP: 93,
  DBP: 80,
  Resp: 16,
  FiO2: 0.21,
  pH: 7.40,
  PaCO2: 40,
  SaO2: 98,
  BUN: 12,
  Calcium: 9.2,
  Glucose: 95,
  Potassium: 4.0,
  Hct: 42,
  Hgb: 14.0,
  WBC: 7.0,
  Platelets: 250,
};

export const UNHEALTHY_ADULT_SAMPLE: Record<string, number> = {
  HR: 125,
  O2Sat: 91,
  Temp: 39.2,
  SBP: 85,
  MAP: 62,
  DBP: 50,
  Resp: 30,
  FiO2: 0.21,
  pH: 7.30,
  PaCO2: 32,
  SaO2: 90,
  BUN: 35,
  Calcium: 7.8,
  Glucose: 180,
  Potassium: 5.2,
  Hct: 32,
  Hgb: 10.5,
  WBC: 18.0,
  Platelets: 90,
};

const COMPARISON_ROWS = [
  { field: 'Heart rate', key: 'HR', healthy: '72', unhealthy: '125', unit: 'bpm', flag: 'Tachycardia' },
  { field: 'Oxygen saturation', key: 'O2Sat', healthy: '98%', unhealthy: '91%', unit: '%', flag: 'Hypoxemia' },
  { field: 'Temperature', key: 'Temp', healthy: '36.8°C', unhealthy: '39.2°C', unit: '°C', flag: 'Fever' },
  { field: 'Systolic pressure', key: 'SBP', healthy: '120', unhealthy: '85', unit: 'mmHg', flag: 'Hypotension' },
  { field: 'Mean pressure (MAP)', key: 'MAP', healthy: '93', unhealthy: '62', unit: 'mmHg', flag: 'Low MAP' },
  { field: 'Diastolic pressure', key: 'DBP', healthy: '80', unhealthy: '50', unit: 'mmHg', flag: 'Low DBP' },
  { field: 'Respiratory rate', key: 'Resp', healthy: '16', unhealthy: '30', unit: '/min', flag: 'Tachypnea' },
  { field: 'FiO2', key: 'FiO2', healthy: '0.21', unhealthy: '0.21', unit: 'fraction', flag: 'Room air' },
  { field: 'pH', key: 'pH', healthy: '7.40', unhealthy: '7.30', unit: '—', flag: 'Acidosis' },
  { field: 'PaCO2', key: 'PaCO2', healthy: '40', unhealthy: '32', unit: 'mmHg', flag: 'Hyperventilation' },
  { field: 'SaO2', key: 'SaO2', healthy: '98%', unhealthy: '90%', unit: '%', flag: 'Hypoxia' },
  { field: 'BUN', key: 'BUN', healthy: '12', unhealthy: '35', unit: 'mg/dL', flag: 'Renal stress' },
  { field: 'Calcium', key: 'Calcium', healthy: '9.2', unhealthy: '7.8', unit: 'mg/dL', flag: 'Hypocalcemia' },
  { field: 'Glucose', key: 'Glucose', healthy: '95', unhealthy: '180', unit: 'mg/dL', flag: 'Hyperglycemia' },
  { field: 'Potassium', key: 'Potassium', healthy: '4.0', unhealthy: '5.2', unit: 'mmol/L', flag: 'Elevated K+' },
  { field: 'Hematocrit', key: 'Hct', healthy: '42%', unhealthy: '32%', unit: '%', flag: 'Reduced Hct' },
  { field: 'Hemoglobin', key: 'Hgb', healthy: '14.0', unhealthy: '10.5', unit: 'g/dL', flag: 'Anemia' },
  { field: 'White blood cells (WBC)', key: 'WBC', healthy: '7.0', unhealthy: '18.0', unit: 'K/µL', flag: 'Leukocytosis' },
  { field: 'Platelets', key: 'Platelets', healthy: '250', unhealthy: '90', unit: 'K/µL', flag: 'Thrombocytopenia' },
];

const measurementNames = [...vitals, ...labs].map(field => field.key);
type FeatureValues = Record<string, number | null>;
type GlobalImportance = { feature: string; mean_abs_shap: number };
type TemporalImportance = {
  feature: string;
  'Early (hour 1-6)'?: number;
  'Mid (hour 7-24)'?: number;
  'Late (hour 25+)'?: number;
};

function generatePatientId(): string {
  const randomNum = Math.floor(100000 + Math.random() * 900000);
  return `p${randomNum}`;
}

export interface StoredPatientAssessment {
  id: string;
  timestamp: string;
  patient_id: string;
  patient_name: string;
  patient_age: number;
  patient_gender: 'Male' | 'Female';
  iculos: number;
  prob_sepsis: number;
  threshold: number;
  prediction: number;
  profileType: 'manual' | 'healthy' | 'unhealthy';
  patientFeatures: FeatureValues;
  observations: FeatureValues;
  shap_top_features?: Array<{ feature: string; shap_value: number; feature_value: number }> | null;
}

function hasObservation(values: FeatureValues | undefined) {
  return Boolean(values && measurementNames.some(name => values[name] !== null && values[name] !== undefined));
}

function getRisk(probability: number, threshold: number) {
  const aboveThreshold = probability >= threshold;
  return {
    aboveThreshold,
    label: aboveThreshold ? 'Above threshold' : 'Below threshold',
    color: probability >= 0.1
      ? 'var(--risk-red)'
      : aboveThreshold ? 'var(--risk-amber)' : 'var(--risk-green)',
  };
}

export default function Prediction() {
  const [patientId, setPatientId] = useState<string>(() => generatePatientId());
  const [patientName, setPatientName] = useState<string>('');
  const [age, setAge] = useState<string>('');
  const [gender, setGender] = useState<'Male' | 'Female' | ''>('');
  const [currentHour, setCurrentHour] = useState(1);
  const [entryHour, setEntryHour] = useState(1);
  const [hourlyHistory, setHourlyHistory] = useState<Record<number, FeatureValues>>({ 1: {} });
  const [patientFeatures, setPatientFeatures] = useState<FeatureValues>({});
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [activeProfile, setActiveProfile] = useState<'manual' | 'healthy' | 'unhealthy'>('manual');
  const [showComparison, setShowComparison] = useState(false);

  // Stored assessments history
  const [historyList, setHistoryList] = useState<StoredPatientAssessment[]>(() => {
    try {
      const saved = localStorage.getItem('sepsis_patient_history');
      if (saved) return JSON.parse(saved);
    } catch {}
    return [];
  });
  const [selectedDetailPatient, setSelectedDetailPatient] = useState<StoredPatientAssessment | null>(null);

  const firstHour = Math.max(1, currentHour - 11);
  const requiredHours = Array.from({ length: currentHour - firstHour + 1 }, (_, index) => firstHour + index);
  const missingHours = requiredHours.filter(hour => !hasObservation(hourlyHistory[hour]));
  const completedHours = requiredHours.length - missingHours.length;
  const activeFeatures = hourlyHistory[entryHour] ?? {};

  const updateObservation = (name: string, value: number | null) => {
    setActiveProfile('manual');
    setHourlyHistory(previous => ({
      ...previous,
      [entryHour]: { ...(previous[entryHour] ?? {}), [name]: value },
    }));
  };

  const applyHealthyProfile = () => {
    setActiveProfile('healthy');
    if (!patientId.trim() || patientId.startsWith('p-sample-')) {
      setPatientId(generatePatientId());
    }
    if (!patientName.trim()) {
      setPatientName('Alex Taylor');
    }
    setAge('45');
    setGender('Male');
    setHourlyHistory(previous => ({
      ...previous,
      [entryHour]: { ...HEALTHY_ADULT_SAMPLE },
    }));
    setPatientFeatures(previous => ({
      ...previous,
      Age: 45,
      Gender: 1,
      Unit1: 0,
      Unit2: 0,
      HospAdmTime: -0.5,
    }));
    setError(null);
    setPrediction(null);
  };

  const applyUnhealthyProfile = () => {
    setActiveProfile('unhealthy');
    if (!patientId.trim() || patientId.startsWith('p-sample-')) {
      setPatientId(generatePatientId());
    }
    if (!patientName.trim()) {
      setPatientName('Morgan Reed');
    }
    setAge('68');
    setGender('Female');
    setHourlyHistory(previous => ({
      ...previous,
      [entryHour]: { ...UNHEALTHY_ADULT_SAMPLE },
    }));
    setPatientFeatures(previous => ({
      ...previous,
      Age: 68,
      Gender: 0,
      Unit1: 0,
      Unit2: 0,
      HospAdmTime: -3.5,
    }));
    setError(null);
    setPrediction(null);
  };

  const applyManualClear = () => {
    setActiveProfile('manual');
    setHourlyHistory(previous => ({
      ...previous,
      [entryHour]: {},
    }));
    setError(null);
    setPrediction(null);
  };

  const selectEntryHour = (hour: number) => {
    setEntryHour(hour);
    setHourlyHistory(previous => ({ ...previous, [hour]: previous[hour] ?? {} }));
  };

  const handleLoadFromHistory = (item: StoredPatientAssessment) => {
    setPatientId(item.patient_id);
    setPatientName(item.patient_name || '');
    setAge(item.patient_age ? String(item.patient_age) : item.patientFeatures?.Age ? String(item.patientFeatures.Age) : '');
    setGender(
      item.patient_gender || (item.patientFeatures?.Gender === 1 ? 'Male' : item.patientFeatures?.Gender === 0 ? 'Female' : '')
    );
    setCurrentHour(item.iculos);
    setEntryHour(item.iculos);
    setHourlyHistory({ [item.iculos]: { ...item.observations } });
    setPatientFeatures({ ...item.patientFeatures });
    setActiveProfile(item.profileType || 'manual');
    setError(null);
    setPrediction(null);
    setSelectedDetailPatient(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleDeleteHistoryItem = (id: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    setHistoryList(prev => {
      const updated = prev.filter(item => item.id !== id);
      try {
        localStorage.setItem('sepsis_patient_history', JSON.stringify(updated));
      } catch {}
      return updated;
    });
    if (selectedDetailPatient?.id === id) {
      setSelectedDetailPatient(null);
    }
  };

  const handleClearHistory = () => {
    if (window.confirm('Clear all stored patient assessment history?')) {
      setHistoryList([]);
      try {
        localStorage.removeItem('sepsis_patient_history');
      } catch {}
      setSelectedDetailPatient(null);
    }
  };

  const runPrediction = async () => {
    if (!patientId.trim()) {
      setError('Patient ID is required (autofilled or entered).');
      return;
    }
    if (!patientName.trim()) {
      setError('Patient name is compulsory. Please enter the patient name.');
      return;
    }
    const numAge = Number(age);
    if (!age || isNaN(numAge) || numAge <= 1) {
      setError('Age is compulsory and must be greater than 1.');
      return;
    }
    if (!gender || (gender !== 'Male' && gender !== 'Female')) {
      setError('Gender is compulsory. Please select Male or Female.');
      return;
    }
    if (missingHours.length) {
      setError(`Add at least one observation for each required hour: ${missingHours.join(', ')}.`);
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const numericGender = gender === 'Male' ? 1 : 0;
      const staticFeatures: Record<string, number> = {
        Age: numAge,
        Gender: numericGender,
        Unit1: typeof patientFeatures.Unit1 === 'number' ? patientFeatures.Unit1 : 0,
        Unit2: typeof patientFeatures.Unit2 === 'number' ? patientFeatures.Unit2 : 0,
        HospAdmTime: typeof patientFeatures.HospAdmTime === 'number' ? patientFeatures.HospAdmTime : -1,
      };

      const request: PredictionRequest = {
        patient_id: patientId.trim(),
        iculos: currentHour,
        history: requiredHours.map(hour => ({
          iculos: hour,
          features: {
            ...staticFeatures,
            ...hourlyHistory[hour],
          },
        })),
      };
      const result = await predictionsApi.create(request);
      setPrediction(result.data);

      // Store in frontend history
      const newRecord: StoredPatientAssessment = {
        id: `assess-${Date.now()}`,
        timestamp: new Date().toLocaleString('en-US', {
          month: 'short', day: 'numeric', year: 'numeric',
          hour: '2-digit', minute: '2-digit'
        }),
        patient_id: patientId.trim(),
        patient_name: patientName.trim(),
        patient_age: numAge,
        patient_gender: gender,
        iculos: currentHour,
        prob_sepsis: result.data.prob_sepsis,
        threshold: result.data.threshold,
        prediction: result.data.prediction,
        profileType: activeProfile,
        patientFeatures: {
          Age: numAge,
          Gender: numericGender,
          ...staticFeatures,
        },
        observations: { ...(hourlyHistory[currentHour] ?? hourlyHistory[entryHour] ?? {}) },
        shap_top_features: result.data.shap_top_features ?? [],
      };
      setHistoryList(prev => {
        const updated = [newRecord, ...prev.filter(item => item.id !== newRecord.id)].slice(0, 50);
        try {
          localStorage.setItem('sepsis_patient_history', JSON.stringify(updated));
        } catch {}
        return updated;
      });
    } catch (requestError) {
      const apiError = requestError as {
        response?: { data?: { detail?: string | Array<{ msg?: string }> } };
        message?: string;
      };
      const detail = apiError.response?.data?.detail;
      setError(
        Array.isArray(detail) ? detail.map(item => item.msg).filter(Boolean).join(' ') :
          detail || apiError.message || 'Prediction could not be completed.'
      );
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setActiveProfile('manual');
    setPatientId(generatePatientId());
    setPatientName('');
    setAge('');
    setGender('');
    setCurrentHour(1);
    setEntryHour(1);
    setHourlyHistory({ 1: {} });
    setPatientFeatures({});
    setPrediction(null);
    setError(null);
  };

  const changeCurrentHour = (value: string) => {
    const nextHour = Math.max(1, Math.min(336, Number(value) || 1));
    setCurrentHour(nextHour);
    setEntryHour(nextHour);
    setPrediction(null);
  };

  const numAgeVal = Number(age);
  const isReady =
    Boolean(patientId.trim()) &&
    Boolean(patientName.trim()) &&
    !isNaN(numAgeVal) &&
    numAgeVal > 1 &&
    (gender === 'Male' || gender === 'Female') &&
    missingHours.length === 0;

  return (
    <div className="prediction-page page-enter">
      <header className="page-heading">
        <div>
          <h1>Sepsis risk prediction</h1>
          <p className="page-subtitle">Enter the recent observations through the current ICU hour.</p>
        </div>
        <button className="btn btn-secondary reset-button" type="button" onClick={resetForm} title="Clear all entered information">
          <RotateCcw size={16} /> Clear
        </button>
      </header>

      <div className="prediction-layout">
        <section className="card prediction-form" aria-labelledby="patient-information-title">
          <div className="card-header">
            <h2 className="card-title" id="patient-information-title">Patient information</h2>
            <span className="form-required">* Required</span>
          </div>

          <div className="patient-identifiers">
            <label className="form-group">
              <span className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Patient ID *</span>
                <button
                  type="button"
                  onClick={() => setPatientId(generatePatientId())}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--color-primary)',
                    cursor: 'pointer',
                    fontSize: '11px',
                    fontWeight: 600,
                    padding: 0,
                    textDecoration: 'underline'
                  }}
                  title="Generate a new random Patient ID"
                >
                  Regenerate ID
                </button>
              </span>
              <input
                className="form-input"
                type="text"
                autoComplete="off"
                placeholder="Autofilled"
                value={patientId}
                onChange={event => setPatientId(event.target.value)}
                required
              />
            </label>
            <label className="form-group">
              <span className="form-label">Current ICU hour *</span>
              <input
                className="form-input"
                type="number"
                min="1"
                max="336"
                value={currentHour}
                onChange={event => changeCurrentHour(event.target.value)}
              />
            </label>
          </div>

          <div className="patient-details-card" style={{
            background: 'rgba(248, 250, 252, 0.85)',
            border: '1px solid var(--color-border)',
            borderRadius: '8px',
            padding: '14px 16px',
            margin: '12px 0 16px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ fontSize: '13px', fontWeight: 650, color: 'var(--color-text)' }}>
                Patient details
              </span>
              <span className="form-required" style={{ fontSize: '11px', color: 'var(--color-primary)' }}>
                * Age &gt; 1 &amp; Gender compulsory
              </span>
            </div>
            <div className="input-grid patient-detail-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
              <label className="form-group">
                <span className="form-label">Patient name *</span>
                <span className="input-unit">Full legal or clinical name</span>
                <input
                  className="form-input"
                  type="text"
                  placeholder="e.g. John Doe"
                  value={patientName}
                  onChange={event => setPatientName(event.target.value)}
                  required
                />
              </label>

              <label className="form-group">
                <span className="form-label">Age *</span>
                <span className="input-unit">years (must be &gt; 1)</span>
                <input
                  className="form-input"
                  type="number"
                  min="2"
                  step="1"
                  placeholder="e.g. 45"
                  value={age}
                  onChange={event => setAge(event.target.value)}
                  required
                />
              </label>

              <label className="form-group">
                <span className="form-label">Gender *</span>
                <span className="input-unit">Biological sex</span>
                <select
                  className="form-input"
                  value={gender}
                  onChange={event => setGender(event.target.value as 'Male' | 'Female' | '')}
                  required
                  style={{ height: '39px', background: 'var(--color-surface)' }}
                >
                  <option value="">Select gender *</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </label>
            </div>
          </div>

          {/* Autofill & Preset Profiles */}
          <div className="autofill-container" aria-label="Input mode and scenario presets">
            <div className="autofill-header">
              <span className="autofill-title">
                <Sparkles size={16} color="var(--color-primary)" />
                Scenario Autofill & Quick Presets
              </span>
              <span className="action-hint">Click a preset to fill values or edit manually</span>
            </div>

            <div className="autofill-buttons">
              <button
                type="button"
                className={`preset-btn preset-manual${activeProfile === 'manual' ? ' active' : ''}`}
                onClick={applyManualClear}
                title="Enter observations manually or clear current hour"
              >
                <div className="preset-btn-top">
                  <Edit3 size={15} />
                  <span>Manual Entry</span>
                  {activeProfile === 'manual' && <span className="preset-active-indicator">Active</span>}
                </div>
                <span className="preset-btn-desc">Type custom observations</span>
              </button>

              <button
                type="button"
                className={`preset-btn preset-healthy${activeProfile === 'healthy' ? ' active' : ''}`}
                onClick={applyHealthyProfile}
                title="Autofill healthy adult reference sample (HR 72, MAP 93, Temp 36.8, WBC 7.0, Platelets 250)"
              >
                <div className="preset-btn-top">
                  <Heart size={15} color="#15803d" />
                  <span>Healthy Adult</span>
                  {activeProfile === 'healthy' && <span className="preset-active-indicator healthy">Loaded</span>}
                </div>
                <span className="preset-btn-desc">Normal vitals & baseline labs (Low Risk)</span>
              </button>

              <button
                type="button"
                className={`preset-btn preset-unhealthy${activeProfile === 'unhealthy' ? ' active' : ''}`}
                onClick={applyUnhealthyProfile}
                title="Autofill unhealthy adult sample (HR 125, MAP 62, Temp 39.2, WBC 18.0, Platelets 90)"
              >
                <div className="preset-btn-top">
                  <AlertTriangle size={15} color="#b91c1c" />
                  <span>Unhealthy (Possible Sepsis)</span>
                  {activeProfile === 'unhealthy' && <span className="preset-active-indicator unhealthy">Loaded</span>}
                </div>
                <span className="preset-btn-desc">Fever, tachycardia, hypotension & sepsis signs</span>
              </button>
            </div>

            <button
              type="button"
              className="comparison-toggle-btn"
              onClick={() => setShowComparison(prev => !prev)}
              aria-expanded={showComparison}
            >
              {showComparison ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              {showComparison ? 'Hide comparison table & clinical note' : 'Quick comparison: Healthy vs. High-risk sample values'}
            </button>

            {showComparison && (
              <div className="comparison-table-wrapper">
                <div className="comparison-table-container">
                  <table className="comparison-table">
                    <thead>
                      <tr>
                        <th>Measurement</th>
                        <th>Healthy sample</th>
                        <th>High-risk sample</th>
                        <th>Unit</th>
                        <th>Clinical significance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {COMPARISON_ROWS.map(row => (
                        <tr key={row.key}>
                          <td><strong>{row.field}</strong></td>
                          <td className="val-healthy">{row.healthy}</td>
                          <td className="val-unhealthy">
                            {row.unhealthy}
                            <span className={['HR', 'Temp', 'Resp', 'BUN', 'Glucose', 'Potassium', 'WBC'].includes(row.key) ? 'val-badge-high' : 'val-badge-low'}>
                              {row.flag}
                            </span>
                          </td>
                          <td>{row.unit}</td>
                          <td style={{ color: 'var(--color-text-muted)' }}>{row.flag}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="synthetic-disclaimer">
                  <Info size={16} />
                  <span>
                    These values represent a synthetic high-risk clinical scenario involving fever, rapid heart rate, low blood pressure, elevated white blood cells, and low platelets. They do not establish that a real person has sepsis.
                  </span>
                </div>
              </div>
            )}
          </div>

          <fieldset className="input-section observation-section">
            <legend>Recent hourly observations</legend>
            <div className="hour-entry-header">
              <div>
                <p className="hour-entry-title">Hour {entryHour}</p>
                <p className="hour-entry-caption">Select each hour and enter available measurements.</p>
              </div>
              <span className="hour-completion">{completedHours} of {requiredHours.length} hours recorded</span>
            </div>
            <div className="hour-selector" role="tablist" aria-label="Recent ICU hours">
              {requiredHours.map(hour => {
                const complete = hasObservation(hourlyHistory[hour]);
                return (
                  <button
                    aria-label={`ICU hour ${hour}${complete ? ', observation entered' : ', observation needed'}`}
                    aria-pressed={entryHour === hour}
                    className={`hour-tab${entryHour === hour ? ' active' : ''}${complete ? ' complete' : ''}`}
                    key={hour}
                    onClick={() => selectEntryHour(hour)}
                    role="tab"
                    type="button"
                  >
                    {complete && <Check size={13} aria-hidden="true" />}
                    {hour}
                  </button>
                );
              })}
            </div>

            <p className="input-group-label">Vitals</p>
            <div className="input-grid">
              {vitals.map(field => (
                <NumericField
                  field={field}
                  key={field.key}
                  value={activeFeatures[field.key]}
                  onChange={value => updateObservation(field.key, value)}
                />
              ))}
            </div>

            <p className="input-group-label labs-label">Laboratory measurements</p>
            <div className="input-grid">
              {labs.map(field => (
                <NumericField
                  field={field}
                  key={field.key}
                  value={activeFeatures[field.key]}
                  onChange={value => updateObservation(field.key, value)}
                />
              ))}
            </div>
            <p className="field-note">Leave unmeasured values blank. The model’s fitted imputer handles missing values.</p>
          </fieldset>

          {error && <div className="form-error" role="alert"><AlertTriangle size={17} /> {error}</div>}

          <div className="form-actions">
            <button className="btn btn-primary predict-action" type="button" onClick={runPrediction} disabled={loading || !isReady}>
              {loading ? <><Loader2 className="loading-icon" size={17} /> Calculating risk…</> : 'Calculate risk'}
            </button>
            <span className="action-hint">Requires one or more observations for each recent hour.</span>
          </div>
        </section>

        <aside className="result-column" aria-live="polite">
          {prediction ? (
            <PredictionResult key={prediction.id} prediction={prediction} />
          ) : (
            <div className="result-empty">
              <h2>Your assessment will appear here</h2>
              <p>Enter the patient’s recent observations to see the calibrated risk estimate, feature drivers, and review prompts.</p>
            </div>
          )}
        </aside>
      </div>

      {/* Patient Assessment History Section */}
      <section className="patient-history-section" aria-labelledby="history-title">
        <div className="history-header">
          <div className="history-title" id="history-title">
            <Clock size={18} color="var(--color-primary)" />
            Patient Assessment History
            <span className="history-count-badge">{historyList.length} recorded</span>
          </div>
          {historyList.length > 0 && (
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: '12px', padding: '4px 10px' }}
              onClick={handleClearHistory}
            >
              <Trash2 size={13} /> Clear History
            </button>
          )}
        </div>

        {historyList.length === 0 ? (
          <div className="history-empty">
            No stored patient assessments yet. Enter values and click <strong>"Calculate risk"</strong> to automatically record and review patient histories here.
          </div>
        ) : (
          <div className="history-table-container">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Patient</th>
                  <th>Age / Sex</th>
                  <th>Date &amp; Time</th>
                  <th>ICU Hour</th>
                  <th>Profile Mode</th>
                  <th>Estimated Sepsis Risk</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {historyList.map(item => {
                  const prob = Math.min(1, Math.max(0, item.prob_sepsis));
                  const risk = getRisk(prob, item.threshold);
                  const displayGender = item.patient_gender || (item.patientFeatures?.Gender === 1 ? 'Male' : item.patientFeatures?.Gender === 0 ? 'Female' : '—');
                  const displayAge = item.patient_age || item.patientFeatures?.Age;
                  return (
                    <tr key={item.id}>
                      <td>
                        <strong>{item.patient_id}</strong>
                        {item.patient_name && (
                          <div style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>{item.patient_name}</div>
                        )}
                      </td>
                      <td>
                        {displayAge ? `${displayAge}y` : '—'} • {displayGender}
                      </td>
                      <td style={{ color: 'var(--color-text-muted)' }}>{item.timestamp}</td>
                      <td>Hour {item.iculos}</td>
                      <td>
                        <span className={`preset-active-indicator ${item.profileType === 'healthy' ? 'healthy' : item.profileType === 'unhealthy' ? 'unhealthy' : ''}`}>
                          {item.profileType === 'healthy' ? 'Healthy Adult' : item.profileType === 'unhealthy' ? 'Unhealthy' : 'Manual'}
                        </span>
                      </td>
                      <td>
                        <strong style={{ color: risk.color }}>{(prob * 100).toFixed(1)}%</strong>
                      </td>
                      <td>
                        <span className={`badge ${risk.aboveThreshold ? 'badge-danger' : 'badge-success'}`}>
                          {risk.label}
                        </span>
                      </td>
                      <td>
                        <div className="history-actions">
                          <button
                            type="button"
                            className="btn-detail"
                            onClick={() => setSelectedDetailPatient(item)}
                            title="View full patient measurements and explainability"
                          >
                            <Eye size={13} /> View details
                          </button>
                          <button
                            type="button"
                            className="btn-load"
                            onClick={() => handleLoadFromHistory(item)}
                            title="Load these observations back into the prediction form"
                          >
                            <RotateCcw size={13} /> Load
                          </button>
                          <button
                            type="button"
                            className="btn-delete-row"
                            onClick={(e) => handleDeleteHistoryItem(item.id, e)}
                            title="Delete this history record"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Full Patient Details Modal */}
      {selectedDetailPatient && (
        <PatientDetailModal
          patient={selectedDetailPatient}
          onClose={() => setSelectedDetailPatient(null)}
          onLoad={() => handleLoadFromHistory(selectedDetailPatient)}
        />
      )}
    </div>
  );
}

type NumericFieldProps = {
  field: { key: string; label: string; unit: string; min: number; max: number; step: number };
  value: number | null | undefined;
  onChange: (value: number | null) => void;
};

function NumericField({ field, value, onChange }: NumericFieldProps) {
  return (
    <label className="form-group observation-field">
      <span className="form-label">{field.label}</span>
      {field.unit && <span className="input-unit">{field.unit}</span>}
      <input
        className="form-input"
        type="number"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value ?? ''}
        onChange={event => onChange(event.target.value === '' ? null : Number(event.target.value))}
      />
    </label>
  );
}

function PredictionResult({ prediction }: { prediction: PredictionResponse }) {
  const [globalImportance, setGlobalImportance] = useState<GlobalImportance[]>([]);
  const [temporalImportance, setTemporalImportance] = useState<TemporalImportance[]>([]);
  const probability = Math.min(1, Math.max(0, prediction.prob_sepsis));
  const risk = getRisk(probability, prediction.threshold);
  const explanations = (prediction.shap_top_features ?? [])
    .filter(item => typeof item.shap_value === 'number')
    .slice(0, 10);

  useEffect(() => {
    let active = true;
    Promise.all([explanationsApi.getGlobalSHAP(), explanationsApi.getTemporalSHAP()])
      .then(([globalResponse, temporalResponse]) => {
        if (active) {
          setGlobalImportance(globalResponse.data);
          setTemporalImportance(temporalResponse.data);
        }
      })
      .catch(() => undefined);
    return () => { active = false; };
  }, []);

  return (
    <div className="result-panel">
      <section className="result-summary" style={{ '--risk-color': risk.color } as React.CSSProperties}>
        <div className="result-heading">
          <div>
            <p className="result-label">Calibrated model estimate</p>
            <p className="result-value">{(probability * 100).toFixed(1)}<span>%</span></p>
          </div>
          <div className="risk-status" style={{ color: risk.color }}>
            {risk.aboveThreshold ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
            {risk.label}
          </div>
        </div>
        <RiskBand probability={probability} threshold={prediction.threshold} animate />
        <p className="threshold-caption">Decision threshold is the alert point selected on validation data; it is not a diagnosis.</p>
      </section>

      <section className="recommendation-section">
        <h2><ShieldAlert size={18} /> Clinician review</h2>
        {risk.aboveThreshold ? (
          <ul>
            <li>Prompt review by the responsible clinician is appropriate.</li>
            <li>Consider the estimate alongside current findings and your local sepsis assessment protocol.</li>
            <li>Verify observations and reassess according to the care team’s plan.</li>
          </ul>
        ) : (
          <ul>
            <li>The estimate is below the selected model threshold.</li>
            <li>Continue observation and reassess if the patient’s condition changes.</li>
            <li>Clinical concern should take precedence over a below-threshold result.</li>
          </ul>
        )}
      </section>

      <section className="explanation-section">
        <div className="explanation-heading">
          <div>
            <h2>What influenced this estimate?</h2>
            <p>Largest CatBoost SHAP drivers for this input</p>
          </div>
          <HelpCircle size={17} aria-label="Positive SHAP values move the CatBoost score toward sepsis; negative values move it away." />
        </div>
        <ShapBars explanations={explanations} animate />
        <p className="explanation-note">
          Positive values move the CatBoost score toward sepsis; negative values move it away. This explains one model in the calibrated ensemble, not a treatment decision.
        </p>
        <details className="global-explainability">
          <summary>Explore global and temporal patterns</summary>
          <p className="explanation-note">Global SHAP shows average feature influence. Temporal SHAP compares importance across ICU stay periods.</p>
          {globalImportance.length > 0 && (
            <div className="global-shap-list">
              <h3>Global feature importance</h3>
              {globalImportance.slice(0, 5).map(item => (
                <div className="global-shap-row" key={item.feature}>
                  <span>{item.feature}</span>
                  <strong>{item.mean_abs_shap.toFixed(3)}</strong>
                </div>
              ))}
            </div>
          )}
          {temporalImportance.length > 0 && (
            <div className="temporal-shap-list">
              <h3>Importance across the ICU stay</h3>
              <div className="table-container">
                <table>
                  <thead><tr><th>Feature</th><th>Early</th><th>Mid</th><th>Late</th></tr></thead>
                  <tbody>
                    {temporalImportance.slice(0, 5).map(item => (
                      <tr key={item.feature}>
                        <td>{item.feature}</td>
                        <td>{item['Early (hour 1-6)']?.toFixed(3) ?? '—'}</td>
                        <td>{item['Mid (hour 7-24)']?.toFixed(3) ?? '—'}</td>
                        <td>{item['Late (hour 25+)']?.toFixed(3) ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {!globalImportance.length && !temporalImportance.length && (
            <p className="explanation-empty">Global explainability artifacts are unavailable.</p>
          )}
        </details>
      </section>

    </div>
  );
}

type PatientDetailModalProps = {
  patient: StoredPatientAssessment;
  onClose: () => void;
  onLoad: () => void;
};

function PatientDetailModal({ patient, onClose, onLoad }: PatientDetailModalProps) {
  const prob = Math.min(1, Math.max(0, patient.prob_sepsis));
  const risk = getRisk(prob, patient.threshold);
  const obs = patient.observations || {};
  const demo = patient.patientFeatures || {};

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="modal-patient-title">
      <div className="modal-card" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title" id="modal-patient-title">
            <FileText size={20} color="var(--color-primary)" />
            <span>Full Patient Details: <strong>{patient.patient_id}</strong></span>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close details">
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {/* Assessment Summary Banner */}
          <div className={`detail-banner ${risk.aboveThreshold ? 'high-risk' : 'low-risk'}`}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Calibrated Sepsis Risk
                </span>
                <span className={`badge ${risk.aboveThreshold ? 'badge-danger' : 'badge-success'}`}>
                  {risk.label}
                </span>
              </div>
              <div style={{ fontSize: '26px', fontWeight: 700 }}>
                {(prob * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: '11px', opacity: 0.85 }}>
                Decision Threshold: {patient.threshold.toFixed(4)} • Model: stacked_ensemble_v1
              </div>
            </div>
            <div style={{ textAlign: 'right', fontSize: '12px' }}>
              <div>Recorded: <strong>{patient.timestamp}</strong></div>
              <div>ICU Stay: <strong>Hour {patient.iculos}</strong></div>
              <div>Profile: <strong>{patient.profileType === 'healthy' ? 'Healthy Adult' : patient.profileType === 'unhealthy' ? 'Unhealthy (Sepsis)' : 'Manual Entry'}</strong></div>
            </div>
          </div>

          {/* Demographics */}
          <div>
            <h3 className="detail-section-title">Patient Demographics & Stay</h3>
            <div className="detail-grid">
              <div className="detail-item">
                <div className="detail-item-label">Patient Name</div>
                <div className="detail-item-val" style={{ fontWeight: 600 }}>{patient.patient_name || '—'}</div>
              </div>
              <div className="detail-item">
                <div className="detail-item-label">Patient ID</div>
                <div className="detail-item-val">{patient.patient_id}</div>
              </div>
              <div className="detail-item">
                <div className="detail-item-label">Age</div>
                <div className="detail-item-val">
                  {patient.patient_age !== undefined && patient.patient_age !== null
                    ? `${patient.patient_age} yrs`
                    : demo.Age !== null && demo.Age !== undefined
                      ? `${demo.Age} yrs`
                      : '—'}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-item-label">Gender</div>
                <div className="detail-item-val">
                  {patient.patient_gender
                    ? patient.patient_gender
                    : demo.Gender === 0
                      ? 'Female'
                      : demo.Gender === 1
                        ? 'Male'
                        : '—'}
                </div>
              </div>
              <div className="detail-item">
                <div className="detail-item-label">ICU Length of Stay</div>
                <div className="detail-item-val">Hour {patient.iculos}</div>
              </div>
            </div>
          </div>

          {/* Vital Signs */}
          <div>
            <h3 className="detail-section-title">Vital Signs Observations</h3>
            <div className="detail-grid">
              {vitals.map(v => {
                const val = obs[v.key];
                return (
                  <div className="detail-item" key={v.key}>
                    <div className="detail-item-label">{v.label} {v.unit ? `(${v.unit})` : ''}</div>
                    <div className="detail-item-val" style={{ color: val !== null && val !== undefined ? 'var(--color-text)' : 'var(--color-text-muted)' }}>
                      {val !== null && val !== undefined ? `${val} ${v.unit}` : 'Not recorded'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Laboratory Measurements */}
          <div>
            <h3 className="detail-section-title">Laboratory Measurements</h3>
            <div className="detail-grid">
              {labs.map(l => {
                const val = obs[l.key];
                return (
                  <div className="detail-item" key={l.key}>
                    <div className="detail-item-label">{l.label} {l.unit ? `(${l.unit})` : ''}</div>
                    <div className="detail-item-val" style={{ color: val !== null && val !== undefined ? 'var(--color-text)' : 'var(--color-text-muted)' }}>
                      {val !== null && val !== undefined ? `${val} ${l.unit}` : 'Not recorded'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Top SHAP Drivers */}
          {patient.shap_top_features && patient.shap_top_features.length > 0 && (
            <div>
              <h3 className="detail-section-title">Top Contributing Feature Drivers (SHAP)</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {patient.shap_top_features.slice(0, 5).map(feat => (
                  <div
                    key={feat.feature}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '7px 12px',
                      background: '#f8fafc',
                      border: '1px solid #e2e8f0',
                      borderRadius: '4px',
                      fontSize: '12px',
                    }}
                  >
                    <span><strong>{feat.feature}</strong>: {feat.feature_value.toFixed(2)}</span>
                    <span style={{
                      fontWeight: 600,
                      color: feat.shap_value > 0 ? '#b91c1c' : '#15803d',
                    }}>
                      {feat.shap_value > 0 ? `+${feat.shap_value.toFixed(3)} (Sepsis risk ↑)` : `${feat.shap_value.toFixed(3)} (Sepsis risk ↓)`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Close
          </button>
          <button type="button" className="btn btn-primary" onClick={onLoad}>
            <RotateCcw size={14} /> Load Observations into Form
          </button>
        </div>
      </div>
    </div>
  );
}