type RiskBandProps = {
  probability: number;
  threshold: number;
  animate?: boolean;
};

export default function RiskBand({ probability, threshold, animate = false }: RiskBandProps) {
  const scaleMaximum = 0.1;
  const markerPosition = Math.min(100, probability / scaleMaximum * 100);
  const thresholdPosition = Math.min(50, threshold / scaleMaximum * 100);
  const riskColor = probability >= 0.1
    ? 'var(--risk-red)'
    : probability >= threshold ? 'var(--risk-amber)' : 'var(--risk-green)';

  return (
    <div className={`risk-band-block${animate ? ' risk-band-animated' : ''}`}>
      <div
        className="risk-scale"
        role="img"
        aria-label={`Estimated risk ${(probability * 100).toFixed(1)} percent; decision threshold ${(threshold * 100).toFixed(2)} percent`}
        style={{
          '--risk-position': `${markerPosition}%`,
          '--threshold-position': `${thresholdPosition}%`,
          '--risk-color': riskColor,
        } as React.CSSProperties}
      >
        <div className="risk-scale-track">
          <span className="risk-zone risk-zone-low" style={{ width: `${thresholdPosition}%` }} />
          <span className="risk-zone risk-zone-watch" style={{ width: `${Math.max(0, 50 - thresholdPosition)}%` }} />
          <span className="risk-zone risk-zone-high" style={{ width: '50%' }} />
        </div>
        <span className="risk-threshold-marker" />
        <span className="risk-value-marker" />
      </div>
      <div className="risk-scale-labels" style={{ '--threshold-position': `${thresholdPosition}%` } as React.CSSProperties}>
        <span>Lower</span>
        <span className="risk-threshold-label">Threshold {(threshold * 100).toFixed(2)}%</span>
        <span>Higher</span>
      </div>
      {probability > scaleMaximum && (
        <p className="threshold-caption">Estimate exceeds this display scale; the numeric value above is exact.</p>
      )}
    </div>
  );
}
