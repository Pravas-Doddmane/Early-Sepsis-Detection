import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { Home, Users, Brain, Activity } from 'lucide-react';
import Overview from './pages/Overview';
import PatientExplorer from './pages/PatientExplorer';
import Explainability from './pages/Explainability';
import Predict from './pages/Predict';

const navigation = [
  { path: '/', label: 'Overview', icon: Home },
  { path: '/patients', label: 'Patients', icon: Users },
  { path: '/explain', label: 'Explainability', icon: Brain },
  { path: '/predict', label: 'Predict', icon: Activity },
];

export default function App() {
  const location = useLocation();

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{
        background: 'var(--color-surface)',
        borderBottom: '1px solid var(--color-border)',
        padding: '16px 24px',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div className="container" style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Link to="/" style={{ fontSize: '20px', fontWeight: '700', color: 'var(--color-primary)', textDecoration: 'none' }}>
            Sepsis Dashboard
          </Link>
          <nav style={{ display: 'flex', gap: '8px' }}>
            {navigation.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '10px 16px',
                  borderRadius: 'var(--radius)',
                  color: location.pathname === item.path ? 'var(--color-primary)' : 'var(--color-text-muted)',
                  background: location.pathname === item.path ? 'rgba(37, 99, 235, 0.1)' : 'transparent',
                  fontWeight: 500,
                  textDecoration: 'none',
                  transition: 'all 0.2s',
                }}
              >
                <item.icon size={18} />
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main style={{ flex: 1, padding: '24px' }}>
        <div className="container">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/patients" element={<PatientExplorer />} />
            <Route path="/explain" element={<Explainability />} />
            <Route path="/explain/:patient_id" element={<Explainability />} />
            <Route path="/predict" element={<Predict />} />
          </Routes>
        </div>
      </main>

      <footer style={{
        background: 'var(--color-surface)',
        borderTop: '1px solid var(--color-border)',
        padding: '16px 24px',
        textAlign: 'center',
        color: 'var(--color-text-muted)',
        fontSize: '13px',
      }}>
        Sepsis Early Warning System — Built with FastAPI + React + TypeScript
      </footer>
    </div>
  );
}
