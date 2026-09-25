import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  HelpCircle,
  Loader2,
  RotateCcw,
  ShieldAlert,
} from 'lucide-react';
import { explanationsApi, PredictionRequest, PredictionResponse, predictionsApi } from '../api';

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

const patientFields = [
  { key: 'Age', label: 'Age', unit: 'years', min: 0, max: 120, step: 1 },
  { key: 'Gender', label: 'Gender code', unit: '0 or 1', min: 0, max: 1, step: 1 },
  { key: 'Unit1', label: 'Hospital unit 1', unit: '0 or 1', min: 0, max: 1, step: 1 },
  { key: 'Unit2', label: 'Hospital unit 2', unit: '0 or 1', min: 0, max: 1, step: 1 },
  { key: 'HospAdmTime', label: 'Admission offset', unit: 'hours', min: -1000, max: 1000, step: 1 },
];

const measurementNames = [...vitals, ...labs].map(field => field.key);
const patientFieldNames = patientFields.map(field => field.key);
type FeatureValues = Record<string, number | null>;
type GlobalImportance = { feature: string; mean_abs_shap: number };
type TemporalImportance = {
  feature: string;
  'Early (hour 1-6)'?: number;
  'Mid (hour 7-24)'?: number;
  'Late (hour 25+)'?: number;
};

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
  const [patientId, setPatientId] = useState('');
  const [currentHour, setCurrentHour] = useState(1);
  const [entryHour, setEntryHour] = useState(1);
  const [hourlyHistory, setHourlyHistory] = useState<Record<number, FeatureValues>>({ 1: {} });
  const [patientFeatures, setPatientFeatures] = useState<FeatureValues>({});
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const firstHour = Math.max(1, currentHour - 11);
  const requiredHours = Array.from({ length: currentHour - firstHour + 1 }, (_, index) => firstHour + index);
  const missingHours = requiredHours.filter(hour => !hasObservation(hourlyHistory[hour]));
  const completedHours = requiredHours.length - missingHours.length;
  const activeFeatures = hourlyHistory[entryHour] ?? {};

  const updateObservation = (name: string, value: number | null) => {
    setHourlyHistory(previous => ({
      ...previous,
      [entryHour]: { ...(previous[entryHour] ?? {}), [name]: value },
    }));
  };

  const selectEntryHour = (hour: number) => {
    setEntryHour(hour);
    setHourlyHistory(previous => ({ ...previous, [hour]: previous[hour] ?? {} }));
  };

  const runPrediction = async () => {
    if (!patientId.trim()) {
      setError('Enter a patient ID to continue.');
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
      const request: PredictionRequest = {
        patient_id: patientId.trim(),
        iculos: currentHour,
        history: requiredHours.map(hour => ({
          iculos: hour,
          features: Object.fromEntries(
            [...measurementNames, ...patientFieldNames].map(name => [
              name,
              patientFieldNames.includes(name)
                ? patientFeatures[name] ?? null
                : hourlyHistory[hour][name] ?? null,
            ])
          ),
        })),
      };
      const result = await predictionsApi.create(request);
      setPrediction(result.data);
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
    setPatientId('');
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

  const isReady = Boolean(patientId.trim()) && missingHours.length === 0;

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
              <span className="form-label">Patient ID *</span>
              <input
                className="form-input"
                type="text"
                autoComplete="off"
                placeholder="e.g. p012345"
                value={patientId}
                onChange={event => setPatientId(event.target.value)}
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

          <details className="input-section patient-details-section">
            <summary>Patient details <span className="optional-label">Optional</span></summary>
            <div className="input-grid patient-detail-grid">
              {patientFields.map(field => (
                <label className="form-group" key={field.key}>
                  <span className="form-label">{field.label}</span>
                  <span className="input-unit">{field.unit}</span>
                  <input
                    className="form-input"
                    type="number"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={patientFeatures[field.key] ?? ''}
                    onChange={event => setPatientFeatures(previous => ({
                      ...previous,
                      [field.key]: event.target.value === '' ? null : Number(event.target.value),
                    }))}
                  />
                </label>
              ))}
            </div>
          </details>

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
  const scaleMaximum = 0.1;
  const markerPosition = Math.min(100, probability / scaleMaximum * 100);
  const thresholdPosition = Math.min(100, prediction.threshold / scaleMaximum * 100);
  const explanations = (prediction.shap_top_features ?? [])
    .filter(item => typeof item.shap_value === 'number')
    .slice(0, 10);
  const maxContribution = Math.max(...explanations.map(item => Math.abs(item.shap_value)), 0.0001);

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
        <div
          className="risk-scale"
          role="img"
          aria-label={`Estimated risk ${(probability * 100).toFixed(1)} percent; decision threshold ${(prediction.threshold * 100).toFixed(2)} percent`}
          style={{
            '--risk-position': `${markerPosition}%`,
            '--threshold-position': `${thresholdPosition}%`,
          } as React.CSSProperties}
        >
          <div className="risk-scale-track">
            <span className="risk-zone risk-zone-low" style={{ width: `${thresholdPosition}%` }} />
            <span className="risk-zone risk-zone-watch" style={{ width: `${Math.max(0, 50 - thresholdPosition)}%` }} />
            <span className="risk-zone risk-zone-high" style={{ width: '50%' }} />
          </div>
          <span className="risk-threshold-marker" />
          <span className="risk-value-marker" style={{ backgroundColor: risk.color }} />
        </div>
        <div className="risk-scale-labels" style={{ '--threshold-position': `${thresholdPosition}%` } as React.CSSProperties}>
          <span>Lower</span>
          <span className="risk-threshold-label">Threshold {(prediction.threshold * 100).toFixed(2)}%</span>
          <span>Higher</span>
        </div>
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
        {explanations.length ? (
          <div className="shap-list">
            {explanations.map((item, index) => (
              <div className="shap-row" key={item.feature}>
                <div className="shap-row-label">
                  <span>{item.feature}</span>
                  <strong className={item.shap_value >= 0 ? 'shap-positive' : 'shap-negative'}>
                    {item.shap_value >= 0 ? '+' : ''}{item.shap_value.toFixed(3)}
                  </strong>
                </div>
                <div className="shap-track">
                  <span
                    className={`shap-bar ${item.shap_value >= 0 ? 'shap-bar-positive' : 'shap-bar-negative'}${item.shap_value >= 0 && Math.abs(item.shap_value) / maxContribution > 0.65 ? ' shap-bar-high' : ''}`}
                    style={{
                      width: `${Math.max(3, Math.abs(item.shap_value) / maxContribution * 100)}%`,
                      animationDelay: `${index * 40}ms`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="explanation-empty">Feature attribution is not available for this prediction.</p>
        )}
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

      <p className="result-disclaimer">Decision support only. Review alongside the full clinical picture.</p>
    </div>
  );
}