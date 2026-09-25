import { Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import { Home as HomeIcon, Activity } from 'lucide-react';
import Home from './pages/Home';
import Prediction from './pages/Prediction';

const navigation = [
  { path: '/', label: 'Home', icon: HomeIcon },
  { path: '/predict', label: 'Prediction', icon: Activity },
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
            Sepsis<span className="brand-secondary"> Dashboard</span>
          </Link>
          <nav style={{ display: 'flex', gap: '8px' }}>
            {navigation.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-link${location.pathname === item.path ? ' active' : ''}`}
                aria-current={location.pathname === item.path ? 'page' : undefined}
              >
                <item.icon size={18} />
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main className="app-main">
        <div className={`container app-page ${location.pathname === '/predict' ? 'app-page-predict' : 'app-page-home'}`}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Prediction />} />
            <Route path="*" element={<Navigate to="/" replace />} />
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
        Sepsis risk decision support · Use with clinical judgment
      </footer>
    </div>
  );
}
