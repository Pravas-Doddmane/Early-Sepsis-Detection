import { SHAPExplanation } from '../api';

type ShapBarsProps = {
  explanations: SHAPExplanation[];
  animate?: boolean;
  limit?: number;
};

export default function ShapBars({ explanations, animate = false, limit = 10 }: ShapBarsProps) {
  const visibleValues = explanations
    .filter(item => typeof item.shap_value === 'number')
    .slice(0, limit);
  const maxContribution = Math.max(...visibleValues.map(item => Math.abs(item.shap_value)), 0.0001);

  if (!visibleValues.length) {
    return <p className="explanation-empty">Feature attribution is not available for this prediction.</p>;
  }

  return (
    <div className={`shap-list${animate ? ' shap-list-animated' : ''}`}>
      {visibleValues.map((item, index) => (
        <div className="shap-row" key={`${item.feature}-${index}`}>
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
                animationDelay: animate ? `${index * 40}ms` : '0ms',
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
