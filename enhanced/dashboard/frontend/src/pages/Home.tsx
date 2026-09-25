import { ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <section className="home-page">
      <div className="home-copy">
        <p className="home-label">Sepsis decision support</p>
        <h1>A clearer view of sepsis risk.</h1>
        <p className="home-intro">
          Review recent ICU observations, see the model’s risk estimate, and understand which signals influenced it.
        </p>
        <Link className="btn btn-primary home-action" to="/predict">
          Start a prediction
        </Link>
        <p className="home-safety">
          <ShieldCheck size={16} /> For clinician review. This tool does not replace clinical judgment or local protocols.
        </p>
      </div>
    </section>
  );
}