import { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, XCircle, Info, Loader2, Stethoscope, Trash2, Plus, Minus } from 'lucide-react';
import { predictionsApi, metricsApi, PredictionRequest, PredictionResponse, ModelMetrics } from '../api';

const inputFields = [
  { key: 'HR', label: 'Heart Rate', unit: 'bpm', min: 30, max: 200, step: 1 },
  { key: 'O2Sat', label: 'O2 Saturation', unit: '%', min: 50, max: 100, step: 1 },
  { key: 'Temp', label: 'Temperature', unit: '°C', min: 30, max: 42, step: 0.1 },
  { key: 'SBP', label: 'Systolic BP', unit: 'mmHg', min: 50, max: 250, step: 1 },
  { key: 'MAP', label: 'Mean Arterial Pressure', unit: 'mmHg', min: 30, max: 150, step: 1 },
  { key: 'DBP', label: 'Diastolic BP', unit: 'mmHg', min: 20, max: 150, step: 1 },
  { key: 'Resp', label: 'Respiratory Rate', unit: 'breaths/min', min: 5, max: 50, step: 1 },
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
  { key: 'WBC', label: 'White Blood Cells', unit: 'K/µL', min: 0, max: 200, step: 0.1 },
  { key: 'Platelets', label: 'Platelets', unit: 'K/µL', min: 0, max: 1500, step: 1 },
];

const staticFields = [
  { key: 'Age', label: 'Age', min: 0, max: 120, step: 1 },
  { key: 'Gender', label: 'Gender code', min: 0, max: 1, step: 1 },
  { key: 'Unit1', label: 'Unit 1 code', min: 0, max: 1, step: 1 },
  { key: 'Unit2', label: 'Unit 2 code', min: 0, max: 1, step: 1 },
  { key: 'HospAdmTime', label: 'Hours from hospital admission', min: -1000, max: 1000, step: 1 },
];

const measurementNames = inputFields.map(field => field.key);
const staticFeatureNames = staticFields.map(field => field.key);

const getRiskLevel = (prob: number, threshold: number) => {
  if (prob >= 0.5) return { level: 'CRITICAL', color: 'var(--color-danger)', icon: AlertTriangle };
  if (prob >= 0.1) return { level: 'HIGH', color: 'var(--color-warning)', icon: AlertTriangle };
  if (prob >= threshold) return { level: 'ELEVATED', color: 'var(--color-warning)', icon: Info };
  return { level: 'LOW', color: 'var(--color-success)', icon: CheckCircle };
};

export default function Predict() {
  const [hourlyHistory, setHourlyHistory] = useState<Record<number, Record<string, number | null>>>({ 1: {} });
  const [staticFeatures, setStaticFeatures] = useState<Record<string, number | null>>(
    Object.fromEntries(staticFeatureNames.map(name => [name, null]))
  );
  const [patientId, setPatientId] = useState('');
  const [iculos, setIculos] = useState(1);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics[]>([]);
  const [threshold, setThreshold] = useState<number | null>(null);
  const features = hourlyHistory[iculos] ?? {};
  const firstRequiredHour = Math.max(1, iculos - 11);
  const requiredHours = Array.from(
    { length: iculos - firstRequiredHour + 1 },
    (_, index) => firstRequiredHour + index
  );
  const missingHours = requiredHours.filter(hour => hourlyHistory[hour] === undefined);

  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  useEffect(() => {
    metricsApi.getModels().then(res => setModelMetrics(res.data)).catch(console.error);
    metricsApi.getOverview().then(res => setThreshold(res.data.metrics.threshold)).catch(console.error);
  }, []);

  const handleFeatureChange = (feature: string, value: number | null) => {
    setHourlyHistory(prev => ({
      ...prev,
      [iculos]: { ...(prev[iculos] ?? {}), [feature]: value },
    }));
  };

  const handleStaticFeatureChange = (feature: string, value: number | null) => {
    setStaticFeatures(prev => ({ ...prev, [feature]: value }));
  };

  const handlePredict = async () => {
    if (!patientId.trim()) {
      setError('Please enter a Patient ID');
      return;
    }
    if (missingHours.length > 0) {
      setError(`Enter measurements for each required ICU hour. Missing: ${missingHours.join(', ')}.`);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const request: PredictionRequest = {
        patient_id: patientId,
        iculos,
        history: requiredHours.map(hour => ({
          iculos: hour,
          features: Object.fromEntries(
            [...measurementNames, ...staticFeatureNames].map(name => [
              name,
              name in staticFeatures ? staticFeatures[name] : hourlyHistory[hour][name] ?? null,
            ])
          ),
        })),
      };
      const res = await predictionsApi.create(request);
      setPrediction(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setHourlyHistory({ 1: {} });
    setStaticFeatures(Object.fromEntries(staticFeatureNames.map(name => [name, null])));
    setIculos(1);
    setPrediction(null);
    setError(null);
  };

  const risk = prediction ? getRiskLevel(prediction.prob_sepsis, prediction.threshold) : null;

  return (
    <div>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px' }}>Sepsis Risk Prediction</h1>
        <p style={{ color: 'var(--color-text-muted)' }}>
          Enter consecutive hourly measurements through the current ICU hour. Threshold:{' '}
          <strong>{threshold === null ? 'Loading' : `${(threshold * 100).toFixed(2)}%`}</strong>.
        </p>
      </div>

      <div className="prediction-layout">
        {/* Input Form */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <div className="card-title">Patient Input</div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
            <div className="form-group">
              <label className="form-label">Patient ID *</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g., p012345"
                value={patientId}
                onChange={e => setPatientId(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Current ICU Hour (ICULOS) *</label>
              <input
                type="number"
                className="form-input"
                min="1"
                max="336"
                value={iculos}
                onChange={e => setIculos(Math.min(336, Math.max(1, parseInt(e.target.value) || 1)))}
              />
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label className="form-label">Hourly Measurements · Hour {iculos}</label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
              {inputFields.map(vital => (
                <div key={vital.key} className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: '12px' }}>
                    {vital.label} ({vital.unit})
                  </label>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ padding: '8px', width: '36px' }}
                      disabled={features[vital.key] == null}
                      onClick={() => handleFeatureChange(vital.key, Math.max(vital.min, (features[vital.key] ?? vital.min) - vital.step))}
                    >
                      <Minus size={14} />
                    </button>
                    <input
                      type="number"
                      className="form-input"
                      min={vital.min}
                      max={vital.max}
                      step={vital.step}
                      value={features[vital.key] ?? ''}
                      onChange={e => handleFeatureChange(vital.key, e.target.value === '' ? null : Number(e.target.value))}
                      style={{ flex: 1 }}
                    />
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ padding: '8px', width: '36px' }}
                      disabled={features[vital.key] == null}
                      onClick={() => handleFeatureChange(vital.key, Math.min(vital.max, (features[vital.key] ?? vital.min) + vital.step))}
                    >
                      <Plus size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label className="form-label">Patient Details</label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '12px' }}>
              {staticFields.map(field => (
                <div key={field.key} className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: '12px' }}>{field.label}</label>
                  <input
                    type="number"
                    className="form-input"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={staticFeatures[field.key] ?? ''}
                    onChange={e => handleStaticFeatureChange(field.key, e.target.value === '' ? null : Number(e.target.value))}
                  />
                </div>
              ))}
            </div>
          </div>

          <p style={{ margin: '0 0 16px', color: 'var(--color-text-muted)', fontSize: '13px' }}>
            Required recent hours: {requiredHours.join(', ')}. Missing lab measurements can be left blank.
          </p>

          <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
            <button
              className="btn btn-primary"
              onClick={handlePredict}
              disabled={loading || !patientId.trim()}
              style={{ flex: 1 }}
            >
              {loading ? (
                <>
                  <Loader2 size={16} style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }} />
                  Predicting...
                </>
              ) : (
                <>
                  <Stethoscope size={16} />
                  Predict Sepsis Risk
                </>
              )}
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              <Trash2 size={16} /> Clear
            </button>
          </div>

          {error && (
            <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(220, 38, 38, 0.1)', border: '1px solid var(--color-danger)', borderRadius: 'var(--radius)', color: 'var(--color-danger)' }}>
              <XCircle size={16} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
              {error}
            </div>
          )}
        </div>

        {/* Result Panel */}
        <div className="card" style={{ height: 'fit-content', position: 'sticky', top: '100px' }}>
          {prediction && risk ? (
            <PredictionResult prediction={prediction} risk={risk} />
          ) : prediction ? (
            <PredictionResult prediction={prediction} risk={getRiskLevel(prediction.prob_sepsis, prediction.threshold)} />
          ) : (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--color-text-muted)' }}>
              <Stethoscope size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
              <p>Enter patient data and click Predict</p>
              <p style={{ fontSize: '13px', marginTop: '8px' }}>
                Causal temporal features are built from the entered hourly history. Missing values use the fitted training imputer.
              </p>
            </div>
          )}

          {/* Model Info */}
          <div style={{ marginTop: '24px', paddingTop: '24px', borderTop: '1px solid var(--color-border)' }}>
            <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '12px' }}>Model Performance</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '13px' }}>
              {modelMetrics.map(m => (
                <div key={m.version} style={{ padding: '12px', background: 'var(--color-bg)', borderRadius: 'var(--radius)' }}>
                  <div style={{ fontWeight: 600, color: m.is_active ? 'var(--color-primary)' : 'var(--color-text)' }}>
                    {m.name} {m.is_active && <span className="badge badge-info" style={{ fontSize: '10px', marginLeft: '6px' }}>Active</span>}
                  </div>
                  <div>ROC-AUC: {m.roc_auc.toFixed(3)}</div>
                  <div>PR-AUC: {m.pr_auc.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PredictionResult({ prediction, risk }: { prediction: PredictionResponse; risk: ReturnType<typeof getRiskLevel> }) {
  const prob = Math.min(1, Math.max(0, prediction.prob_sepsis));
  const threshold = prediction.threshold;
  const gaugeCircumference = 2 * Math.PI * 80;

  return (
    <div>
      {/* Risk Gauge */}
      <div style={{ textAlign: 'center', marginBottom: '24px' }}>
        <div style={{ fontSize: '14px', color: 'var(--color-text-muted)', marginBottom: '8px' }}>
          SEPSIS PROBABILITY
        </div>
        <div style={{ position: 'relative', width: '200px', height: '200px', margin: '0 auto' }}>
          <svg viewBox="0 0 200 200" style={{ transform: 'rotate(-90deg)' }}>
            <circle
              cx="100" cy="100" r="80"
              fill="none"
              stroke="var(--color-border)"
              strokeWidth="16"
            />
            <circle
              cx="100" cy="100" r="80"
              fill="none"
              stroke={risk.color}
              strokeWidth="16"
              strokeDasharray={`${prob * gaugeCircumference} ${gaugeCircumference}`}
              strokeLinecap="round"
              style={{ transition: 'stroke-dasharray 0.5s ease' }}
            />
            {/* Threshold marker */}
            <line
              x1="100" y1="20"
              x2="100" y2="36"
              stroke="var(--color-text-muted)"
              strokeWidth="2"
              style={{ transform: `rotate(${threshold * 100 * 3.6}deg)`, transformOrigin: '100 100' }}
            />
          </svg>
          <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
            <div style={{ fontSize: '42px', fontWeight: '700', color: risk.color }}>
              {(prob * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '12px', color: 'var(--color-text-muted)' }}>
              Threshold: {(threshold * 100).toFixed(2)}%
            </div>
          </div>
        </div>

        <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'center', gap: '16px', flexWrap: 'wrap' }}>
          <span
            className="badge"
            style={{
              background: risk.color,
              color: 'white',
              fontSize: '14px',
              padding: '8px 16px',
            }}
          >
            <risk.icon size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
            {risk.level} RISK
          </span>
          <span className="badge badge-info">
            ICU Hour: {prediction.iculos}
          </span>
        </div>

        {prediction.true_label !== null && (
          <div style={{ marginTop: '16px', padding: '12px', background: 'var(--color-bg)', borderRadius: 'var(--radius)', fontSize: '13px' }}>
            <strong>Ground Truth: </strong>
            <span className={`badge ${prediction.true_label === 1 ? 'badge-danger' : 'badge-success'}`} style={{ marginLeft: '8px' }}>
              {prediction.true_label === 1 ? 'Sepsis' : 'No Sepsis'}
            </span>
            {prediction.prediction === prediction.true_label ? (
              <span style={{ marginLeft: '12px', color: 'var(--color-success)' }}>
                <CheckCircle size={14} style={{ verticalAlign: 'middle' }} /> Correct
              </span>
            ) : (
              <span style={{ marginLeft: '12px', color: 'var(--color-danger)' }}>
                <XCircle size={14} style={{ verticalAlign: 'middle' }} /> Incorrect
              </span>
            )}
          </div>
        )}
      </div>

      {/* Contribution Breakdown */}
      <div>
        <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '4px' }}>Ensemble Model Scores</h3>
        <p style={{ fontSize: '12px', color: 'var(--color-text-muted)', marginBottom: '12px' }}>
          These are base-model probabilities, not patient-level feature contributions.
        </p>
        <div className="table-container" style={{ maxHeight: '300px' }}>
          <table style={{ fontSize: '13px' }}>
            <thead>
              <tr>
                <th>Model</th>
                <th>Probability</th>
                <th>Output</th>
              </tr>
            </thead>
            <tbody>
              {prediction.shap_top_features?.slice(0, 10).map((item: any, idx: number) => (
                <tr key={idx}>
                  <td style={{ fontFamily: 'monospace', fontSize: '12px' }}>{item.feature || item.model}</td>
                  <td>{item.feature_value?.toFixed(2) ?? item.probability?.toFixed(3) ?? '—'}</td>
                  <td>
                    {item.shap_value !== undefined ? (
                      <span className={`badge ${item.shap_value >= 0 ? 'badge-danger' : 'badge-success'}`} style={{ fontSize: '11px' }}>
                        {item.shap_value >= 0 ? '↑' : '↓'} {Math.abs(item.shap_value).toFixed(3)}
                      </span>
                    ) : item.probability ? (
                      <span className="badge badge-info" style={{ fontSize: '11px' }}>
                        {item.model}: {(item.probability * 100).toFixed(1)}%
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Clinical Action */}
      <div style={{ marginTop: '24px', padding: '16px', background: 'var(--color-bg)', borderRadius: 'var(--radius)', border: `1px solid ${risk.color}` }}>
        <h4 style={{ marginBottom: '8px', color: risk.color, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <risk.icon size={18} /> Clinical Action
        </h4>
        <ul style={{ fontSize: '13px', color: 'var(--color-text-muted)', paddingLeft: '20px', lineHeight: '1.8' }}>
          {prob >= threshold ? (
            <>
              <li>Initiate sepsis protocol: lactate, blood cultures, broad-spectrum antibiotics within 1 hour</li>
              <li>Fluid resuscitation: 30 mL/kg crystalloid for hypotension/lactate ≥4</li>
              <li>Vasopressors if MAP {'<'}65 after fluids</li>
              <li>Reassess in 1 hour — monitor SOFA/qSOFA</li>
            </>
          ) : (
            <>
              <li>Routine monitoring per unit protocol</li>
              <li>Re-screen at next ICU hour or with clinical change</li>
            </>
          )}
        </ul>
      </div>
    </div>
  );
}
