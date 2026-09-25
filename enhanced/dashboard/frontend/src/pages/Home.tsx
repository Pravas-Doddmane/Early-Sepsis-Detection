import { ArrowRight, Activity, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <section className="home-page page-enter">
      <div className="home-copy">
        <p className="eyebrow"><Activity size={15} /> CLINICAL DECISION SUPPORT</p>
        <h1>A clearer view of sepsis risk.</h1>
        <p className="home-intro">
          Review recent ICU observations, see the model’s risk estimate, and understand which signals influenced it.
        </p>
        <Link className="btn btn-primary home-action" to="/predict">
          Start a prediction <ArrowRight size={17} />
        </Link>
        <p className="home-safety">
          <ShieldCheck size={16} /> For clinician review. This tool does not replace clinical judgment or local protocols.
        </p>
      </div>
      <div className="home-mark" aria-hidden="true">
        <div className="home-mark-line" />
        <Activity size={52} strokeWidth={1.4} />
      </div>
    </section>
  );
}