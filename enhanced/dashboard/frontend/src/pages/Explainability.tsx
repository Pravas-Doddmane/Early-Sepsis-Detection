import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line } from 'recharts';
import { ArrowLeft, Image } from 'lucide-react';
import { explanationsApi } from '../api';

type Tab = 'patient' | 'global' | 'temporal';

const COLORS = ['#2563eb', '#16a34a', '#f59e0b', '#dc2626', '#9333ea', '#ec4899', '#06b6d4', '#84cc16'];

export default function Explainability() {
  const params = useParams<{ patient_id?: string }>();
  const [activeTab, setActiveTab] = useState<Tab>('patient');
  const [globalSHAP, setGlobalSHAP] = useState<Array<{ feature: string; mean_abs_shap: number }>>([]);
  const [temporalSHAP, setTemporalSHAP] = useState<Array<{ feature: string; 'Early (hour 1-6)': number; 'Mid (hour 7-24)': number; 'Late (hour 25+)': number }>>([]);
  const [waterfallLabel, setWaterfallLabel] = useState<'TP' | 'TN'>('TP');
  const [limeLabel, setLimeLabel] = useState<'TP' | 'TN'>('TP');
  const [waterfallError, setWaterfallError] = useState<string | null>(null);
  const [limeError, setLimeError] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ src: string; title: string } | null>(null);

  useEffect(() => {
    if (params.patient_id) {
      setActiveTab('patient');
    }
    loadGlobalSHAP();
    loadTemporalSHAP();
  }, [params.patient_id]);

  const loadGlobalSHAP = async () => {
    try {
      const res = await explanationsApi.getGlobalSHAP();
      setGlobalSHAP(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const loadTemporalSHAP = async () => {
    try {
      const res = await explanationsApi.getTemporalSHAP();
      setTemporalSHAP(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  // Top features for charts
  const topSHAP = globalSHAP.slice(0, 15).reverse();
  const topTemporal = temporalSHAP.slice(0, 8);
  const temporalChartData = ['Early (hour 1-6)', 'Mid (hour 7-24)', 'Late (hour 25+)'].map(period => ({
    period,
    ...Object.fromEntries(topTemporal.map(f => [f.feature, f[period as keyof typeof f] || 0]))
  }));

  return (
    <div>
      {/* Back link if in patient context */}
      {params.patient_id && (
        <Link to="/patients" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '16px', color: 'var(--color-text-muted)' }}>
          <ArrowLeft size={16} /> Back to Patients
        </Link>
      )}

      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px' }}>Explainability</h1>
        <p style={{ color: 'var(--color-text-muted)' }}>
          SHAP (global + temporal) and LIME explanations for the CatBoost base model.
        </p>
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'patient' ? 'active' : ''}`}
          onClick={() => setActiveTab('patient')}
        >
          Patient-Level
        </button>
        <button
          className={`tab ${activeTab === 'global' ? 'active' : ''}`}
          onClick={() => setActiveTab('global')}
        >
          Global Importance
        </button>
        <button
          className={`tab ${activeTab === 'temporal' ? 'active' : ''}`}
          onClick={() => setActiveTab('temporal')}
        >
          Temporal Shift
        </button>
      </div>

      {/* Patient-Level Tab */}
      {activeTab === 'patient' && (
        <div>
          {params.patient_id ? (
            <PatientExplanationView
              waterfallLabel={waterfallLabel}
              setWaterfallLabel={setWaterfallLabel}
              limeLabel={limeLabel}
              setLimeLabel={setLimeLabel}
              waterfallError={waterfallError}
              setWaterfallError={setWaterfallError}
              limeError={limeError}
              setLimeError={setLimeError}
              openPreview={setPreview}
            />
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '60px 20px' }}>
              <p style={{ color: 'var(--color-text-muted)', marginBottom: '16px' }}>
                Select a patient from the <Link to="/patients">Patient Explorer</Link> to view per-patient explanations.
              </p>
              <p style={{ fontSize: '14px', color: 'var(--color-text-muted)' }}>
                Or view the pre-computed example cases below (TP/TN from test set).
              </p>
              <ExampleCases />
            </div>
          )}
        </div>
      )}

      {/* Global Importance Tab */}
      {activeTab === 'global' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Global Feature Importance (SHAP Beeswarm)</div>
            <span className="badge badge-info">Top 15 of {globalSHAP.length}</span>
          </div>
          <div style={{ height: '500px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topSHAP} layout="vertical" margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={200}
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(4), 'Mean |SHAP|']}
                />
                <Bar dataKey="mean_abs_shap" radius={[0, 4, 4, 0]} barSize={28}>
                  {topSHAP.map((_, idx) => <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', gap: '16px', marginTop: '16px', justifyContent: 'center' }}>
            <button className="btn btn-secondary" onClick={() => setPreview({ src: explanationsApi.getGlobalSHAPImage(), title: 'Global SHAP beeswarm plot' })}>
              <Image size={16} /> View Full Beeswarm Plot
            </button>
          </div>
        </div>
      )}

      {/* Temporal Shift Tab */}
      {activeTab === 'temporal' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Temporal Feature Importance Shift</div>
            <span className="badge badge-info">Novel vs. static models</span>
          </div>
          <div style={{ height: '450px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={temporalChartData} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(4), 'Mean |SHAP|']}
                />
                {topTemporal.map((f, idx) => (
                  <Line
                    key={f.feature}
                    type="monotone"
                    dataKey={f.feature}
                    stroke={COLORS[idx % COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 4, strokeWidth: 2 }}
                    activeDot={{ r: 6 }}
                    name={f.feature}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', gap: '16px', marginTop: '16px', justifyContent: 'center' }}>
            <button className="btn btn-secondary" onClick={() => setPreview({ src: explanationsApi.getTemporalSHAPImage(), title: 'Temporal feature importance shift' })}>
              <Image size={16} /> View Full Plot
            </button>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--color-text-muted)', marginTop: '16px', textAlign: 'center' }}>
            Shows how feature importance changes across ICU stay phases — impossible with static stay-level models.
          </p>
        </div>
      )}

      {preview && <ImagePreview {...preview} onClose={() => setPreview(null)} />}
    </div>
  );
}

// Sub-component for patient explanation view
function PatientExplanationView({
  waterfallLabel,
  setWaterfallLabel,
  limeLabel,
  setLimeLabel,
  waterfallError,
  setWaterfallError,
  limeError,
  setLimeError,
  openPreview,
}: {
  waterfallLabel: 'TP' | 'TN';
  setWaterfallLabel: (v: 'TP' | 'TN') => void;
  limeLabel: 'TP' | 'TN';
  setLimeLabel: (v: 'TP' | 'TN') => void;
  waterfallError: string | null;
  setWaterfallError: (v: string | null) => void;
  limeError: string | null;
  setLimeError: (v: string | null) => void;
  openPreview: (preview: { src: string; title: string }) => void;
}) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          Representative Example Cases
        </div>
      </div>

      <p style={{ color: 'var(--color-text-muted)', fontSize: '13px', marginBottom: '20px' }}>
        These plots explain one representative true-positive case and one representative true-negative case.
        They are not aggregate plots and do not represent every patient.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* SHAP Waterfall */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: '600' }}>SHAP Waterfall</h3>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                className={`btn ${waterfallLabel === 'TP' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => { setWaterfallLabel('TP'); setWaterfallError(null); }}
              >
                True Positive Case
              </button>
              <button
                className={`btn ${waterfallLabel === 'TN' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => { setWaterfallLabel('TN'); setWaterfallError(null); }}
              >
                True Negative Case
              </button>
            </div>
          </div>
          {waterfallError && (
            <div style={{ color: 'var(--color-danger)', fontSize: '13px', marginBottom: '12px' }}>{waterfallError}</div>
          )}
          <div style={{ border: '1px solid var(--color-border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
            <img
              src={explanationsApi.getPatientWaterfallImage(waterfallLabel)}
              alt={`SHAP Waterfall ${waterfallLabel}`}
              style={{ width: '100%', height: 'auto', display: 'block' }}
              onError={(e) => {
                e.currentTarget.style.display = 'none';
                setWaterfallError('Image not found. Run XAI script to generate.');
              }}
              onClick={() => openPreview({ src: explanationsApi.getPatientWaterfallImage(waterfallLabel), title: `SHAP waterfall — ${waterfallLabel} example` })}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => event.key === 'Enter' && openPreview({ src: explanationsApi.getPatientWaterfallImage(waterfallLabel), title: `SHAP waterfall — ${waterfallLabel} example` })}
            />
            {waterfallError && (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                <Image size={32} style={{ marginBottom: '8px', opacity: 0.5 }} />
                <p>Waterfall plot for {waterfallLabel} example</p>
                <p style={{ fontSize: '12px' }}>Run: python enhanced/xai/explain.py</p>
              </div>
            )}
          </div>
        </div>

        {/* LIME Explanation */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: '600' }}>LIME Local Explanation</h3>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                className={`btn ${limeLabel === 'TP' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => { setLimeLabel('TP'); setLimeError(null); }}
              >
                True Positive Case
              </button>
              <button
                className={`btn ${limeLabel === 'TN' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => { setLimeLabel('TN'); setLimeError(null); }}
              >
                True Negative Case
              </button>
            </div>
          </div>
          {limeError && (
            <div style={{ color: 'var(--color-danger)', fontSize: '13px', marginBottom: '12px' }}>{limeError}</div>
          )}
          <div style={{ border: '1px solid var(--color-border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
            <img
              src={explanationsApi.getPatientLIMEImage(limeLabel)}
              alt={`LIME ${limeLabel}`}
              style={{ width: '100%', height: 'auto', display: 'block' }}
              onError={(e) => {
                e.currentTarget.style.display = 'none';
                setLimeError('Image not found. Run XAI script to generate.');
              }}
              onClick={() => openPreview({ src: explanationsApi.getPatientLIMEImage(limeLabel), title: `LIME explanation — ${limeLabel} example` })}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => event.key === 'Enter' && openPreview({ src: explanationsApi.getPatientLIMEImage(limeLabel), title: `LIME explanation — ${limeLabel} example` })}
            />
            {limeError && (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                <Image size={32} style={{ marginBottom: '8px', opacity: 0.5 }} />
                <p>LIME plot for {limeLabel} example</p>
                <p style={{ fontSize: '12px' }}>Run: python enhanced/xai/explain.py</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Pre-computed example cases
function ImagePreview({ src, title, onClose }: { src: string; title: string; onClose: () => void }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onClick={onClose}
      style={{ position: 'fixed', inset: 0, zIndex: 1000, padding: '32px', background: 'rgba(15, 23, 42, 0.9)', display: 'flex', flexDirection: 'column', alignItems: 'center', overflow: 'auto' }}
    >
      <div style={{ width: 'min(1200px, 100%)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'white', marginBottom: '12px' }}>
        <strong>{title}</strong>
        <button type="button" className="btn btn-secondary" onClick={onClose}>Close</button>
      </div>
      <img src={src} alt={title} onClick={(event) => event.stopPropagation()} style={{ maxWidth: '1200px', width: '100%', height: 'auto', background: 'white', borderRadius: 'var(--radius)' }} />
    </div>
  );
}

function ExampleCases() {
  return (
    <div className="card" style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div className="card-header">
          <div className="card-title">Representative Example Cases</div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div style={{ textAlign: 'center', padding: '24px', background: 'var(--color-bg)', borderRadius: 'var(--radius)' }}>
            <h4 style={{ marginBottom: '12px', color: 'var(--color-danger)' }}>Representative True Positive</h4>
          <p style={{ fontSize: '13px', color: 'var(--color-text-muted)', marginBottom: '16px' }}>
            Correctly predicted sepsis case
          </p>
          <a className="btn btn-secondary" href={explanationsApi.getPatientWaterfallImage('TP')} target="_blank" rel="noreferrer">
            <Image size={16} /> View SHAP Waterfall
          </a>
        </div>
        <div style={{ textAlign: 'center', padding: '24px', background: 'var(--color-bg)', borderRadius: 'var(--radius)' }}>
            <h4 style={{ marginBottom: '12px', color: 'var(--color-success)' }}>Representative True Negative</h4>
          <p style={{ fontSize: '13px', color: 'var(--color-text-muted)', marginBottom: '16px' }}>
            Correctly predicted non-sepsis case
          </p>
          <a className="btn btn-secondary" href={explanationsApi.getPatientWaterfallImage('TN')} target="_blank" rel="noreferrer">
            <Image size={16} /> View SHAP Waterfall
          </a>
        </div>
      </div>
    </div>
  );
}
