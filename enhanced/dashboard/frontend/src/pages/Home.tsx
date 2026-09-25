import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="home-content">
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
        </div>
        <div className="home-waveform" aria-hidden="true">
          <svg viewBox="0 0 400 100" focusable="false">
            <path
              className="ecg-path"
              d="M0 50 H55 L68 50 L80 20 L92 78 L105 50 H155 L168 50 L180 12 L193 86 L207 50 H255 L268 50 L280 25 L292 75 L305 50 H355 L368 50 L380 16 L393 82 L400 50"
            />
          </svg>
        </div>
      </section>

      <section className="home-how" aria-labelledby="how-heading">
        <h2 id="how-heading">How it works</h2>
        <ol className="home-steps">
          <li>
            <span className="step-number">1</span>
            <div><h3>Enter recent observations</h3><p>Vitals and labs from the current ICU hour.</p></div>
          </li>
          <li>
            <span className="step-number">2</span>
            <div><h3>Model estimates risk</h3><p>Calibrated probability compared with a clinical threshold.</p></div>
          </li>
          <li>
            <span className="step-number">3</span>
            <div><h3>Review what drove it</h3><p>Feature importance, in plain terms.</p></div>
          </li>
        </ol>
      </section>

      <section className="home-timing" aria-labelledby="timing-heading">
        <h2 id="timing-heading">Why timing matters</h2>
        <p>Sepsis is a life-threatening medical emergency. The CDC advises immediate evaluation when sepsis is suspected.</p>
        <p>Without fast treatment, sepsis can quickly lead to tissue damage, organ failure, and death.</p>
        <a href="https://www.cdc.gov/sepsis/about/index.html" target="_blank" rel="noreferrer">
          Source: CDC, About Sepsis
        </a>
      </section>
    </div>
  );
}