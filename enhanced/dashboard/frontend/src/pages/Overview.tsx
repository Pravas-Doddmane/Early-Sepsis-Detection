import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line } from 'recharts';
import { CheckCircle, Database, Shield } from 'lucide-react';
import { metricsApi, OverviewResponse } from '../api';

export default function Overview() {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    metricsApi.getOverview()
      .then(res => setData(res.data))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="card">Loading...</div>;
  if (error) return <div className="card" style={{ color: 'var(--color-danger)' }}>Error: {error}</div>;
  if (!data) return <div className="card">No data</div>;

  const { metrics, feature_importance, temporal_importance } = data;

  // Top 10 features for chart
  const topFeatures = feature_importance.slice(0, 10).reverse();

  // Temporal data for top 5 features
  const topTemporalFeatures = temporal_importance.slice(0, 5);
  const temporalChartData = ['Early (hour 1-6)', 'Mid (hour 7-24)', 'Late (hour 25+)'].map((period) => ({
    period,
    ...Object.fromEntries(
      topTemporalFeatures.map(f => [f.feature, f[period as keyof typeof f] || 0])
    )
  }));

  const COLORS = ['#2563eb', '#16a34a', '#f59e0b', '#dc2626', '#9333ea', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <div>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px' }}>Model Overview</h1>
        <p style={{ color: 'var(--color-text-muted)' }}>
          Stacked ensemble (CatBoost + XGBoost + LightGBM + RF) with isotonic calibration.
          Temporal features + leak-safe OOF stacking + external validation.
        </p>
      </div>

      {/* Key Metrics Grid */}
      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">Test ROC-AUC</div>
          <div className="metric-value success">{metrics.test_roc_auc.toFixed(3)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Test PR-AUC</div>
          <div className="metric-value">{metrics.test_pr_auc.toFixed(3)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Sensitivity (at threshold)</div>
          <div className="metric-value success">{(metrics.test_sensitivity * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Specificity</div>
          <div className="metric-value">{(metrics.test_specificity * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Precision</div>
          <div className="metric-value">{(metrics.test_precision * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">F1 Score</div>
          <div className="metric-value">{metrics.test_f1.toFixed(3)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">MCC</div>
          <div className="metric-value">{metrics.test_mcc.toFixed(3)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Threshold</div>
          <div className="metric-value">{metrics.threshold.toFixed(4)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Calibration ECE</div>
          <div className="metric-value success">{metrics.calibration_ece.toFixed(4)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Ext. Val Set A ROC</div>
          <div className="metric-value">{metrics.external_setA_roc.toFixed(3)}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Ext. Val Set B ROC</div>
          <div className="metric-value">{metrics.external_setB_roc.toFixed(3)}</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '24px' }}>
        {/* Global Feature Importance */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Global Feature Importance (SHAP)</div>
            <span className="badge badge-info">Top 10 of 150</span>
          </div>
          <div style={{ height: '400px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
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
                  labelFormatter={(label: string) => label}
                />
                <Bar
                  dataKey="mean_abs_shap"
                  radius={[0, 4, 4, 0]}
                  barSize={28}
                >
                  {topFeatures.map((_, i) => (
                    <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--color-text-muted)', marginTop: '12px', textAlign: 'center' }}>
            Mean absolute SHAP values across 500 sampled patients. Higher = more impact on predictions.
          </p>
        </div>

        {/* Temporal Feature Importance */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Temporal Importance Shift</div>
            <span className="badge badge-info">Novel vs. static models</span>
          </div>
          <div style={{ height: '400px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={temporalChartData} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(4), 'Mean |SHAP|']}
                  labelFormatter={(label: string) => label}
                />
                <Tooltip />
                {topTemporalFeatures.map((f, i) => (
                  <Line
                    key={f.feature}
                    type="monotone"
                    dataKey={f.feature}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={2}
                    dot={{ r: 4, strokeWidth: 2 }}
                    activeDot={{ r: 6 }}
                    name={f.feature}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--color-text-muted)', marginTop: '12px', textAlign: 'center' }}>
            How feature importance shifts across ICU stay phases. Static models cannot show this.
          </p>
        </div>
      </div>

      {/* Validation & Quality Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginTop: '24px' }}>
        <div className="card">
          <div className="card-header">
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Shield className="success" />
              External Validation (Holdout Sets)
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ padding: '16px', background: 'var(--color-bg)', borderRadius: 'var(--radius)' }}>
              <div style={{ fontWeight: '600', marginBottom: '8px', color: 'var(--color-primary)' }}>Set A</div>
              <div style={{ fontSize: '13px', color: 'var(--color-text-muted)' }}>
                <div>Patients: 3,036</div>
                <div>Sepsis rate: 2.1%</div>
                <div><strong>ROC-AUC: {metrics.external_setA_roc.toFixed(3)}</strong></div>
                <div>Sensitivity: 72.6%</div>
              </div>
            </div>
            <div style={{ padding: '16px', background: 'var(--color-bg)', borderRadius: 'var(--radius)' }}>
              <div style={{ fontWeight: '600', marginBottom: '8px', color: 'var(--color-primary)' }}>Set B</div>
              <div style={{ fontSize: '13px', color: 'var(--color-text-muted)' }}>
                <div>Patients: 3,015</div>
                <div>Sepsis rate: 1.5%</div>
                <div><strong>ROC-AUC: {metrics.external_setB_roc.toFixed(3)}</strong></div>
                <div>Sensitivity: 55.4%</div>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Database />
              Data Quality
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', fontSize: '13px' }}>
            <div><strong>Total Patients:</strong> 40,336</div>
            <div><strong>Sepsis Rate:</strong> 7.3%</div>
            <div><strong>Median ICU Stay:</strong> 38 hours</div>
            <div><strong>Total Hourly Records:</strong> 1.55M</div>
            <div><strong>Lab Missingness:</strong> {'>'}80% avg</div>
            <div><strong>Vitals Missingness:</strong> {'<'}15% avg</div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle className="success" />
              Pipeline Quality Checks
            </div>
          </div>
          <ul style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', listStyle: 'none', fontSize: '14px' }}>
            <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle className="success" size={16} /> Leak-safe OOF stacking</li>
            <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle className="success" size={16} /> Patient-level splits</li>
            <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle className="success" size={16} /> Isotonic calibration</li>
            <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle className="success" size={16} /> Clinical threshold tuning</li>
            <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle className="success" size={16} /> External validation</li>
            <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle className="success" size={16} /> SHAP + LIME explainability</li>
          </ul>
        </div>
      </div>
    </div>
  );
}