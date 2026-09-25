import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface Patient {
  patient_id: string;
  age: number | null;
  gender: number | null;
  unit1: number | null;
  unit2: number | null;
  hosp_adm_time: number | null;
  sepsis_label: number | null;
  onset_hour: number | null;
}

export interface PatientListResponse {
  patients: Patient[];
  total: number;
  page: number;
  page_size: number;
}

export interface HourlyRecord {
  id: number;
  patient_id: string;
  iculos: number;
  hr: number | null;
  o2sat: number | null;
  temp: number | null;
  sbp: number | null;
  map: number | null;
  dbp: number | null;
  resp: number | null;
  sepsis_label: number | null;
  // ... other fields
}

export interface PredictionRequest {
  patient_id: string;
  iculos: number;
  features: Record<string, number>;
}

export interface PredictionResponse {
  id: number;
  patient_id: string;
  iculos: number;
  model_version: string;
  prob_sepsis: number;
  threshold: number;
  prediction: number;
  true_label: number | null;
  shap_top_features: any[] | null;
  created_at: string;
}

export interface SHAPExplanation {
  feature: string;
  shap_value: number;
  feature_value: number;
}

export interface LIMEExplanation {
  feature: string;
  weight: number;
}

export interface ExplanationResponse {
  patient_id: string;
  iculos: number;
  shap: SHAPExplanation[];
  lime: LIMEExplanation[];
  prediction_prob: number;
  prediction: number;
  true_label: number | null;
}

export interface ModelMetrics {
  version: string;
  name: string;
  roc_auc: number;
  pr_auc: number;
  threshold: number;
  calibration_type: string;
  is_active: boolean;
  created_at: string | null;
}

export interface OverviewMetrics {
  test_roc_auc: number;
  test_pr_auc: number;
  test_sensitivity: number;
  test_specificity: number;
  test_precision: number;
  test_f1: number;
  test_mcc: number;
  threshold: number;
  calibration_ece: number;
  external_setA_roc: number;
  external_setB_roc: number;
}

export interface OverviewResponse {
  metrics: OverviewMetrics;
  feature_importance: Array<{ feature: string; mean_abs_shap: number }>;
  temporal_importance: Array<{
    feature: string;
    'Early (hour 1-6)': number;
    'Mid (hour 7-24)': number;
    'Late (hour 25+)': number;
  }>;
}

export interface ExternalValidation {
  setA: {
    n_rows: number;
    n_patients: number;
    sepsis_rate: number;
    roc_auc: number;
    pr_auc: number;
    sensitivity: number;
    precision: number;
  };
  setB: {
    n_rows: number;
    n_patients: number;
    sepsis_rate: number;
    roc_auc: number;
    pr_auc: number;
    sensitivity: number;
    precision: number;
  };
}

// API functions
export const patientsApi = {
  list: (params?: { page?: number; page_size?: number; sepsis_only?: boolean }) =>
    api.get<PatientListResponse>('/patients', { params }),

  get: (patient_id: string) =>
    api.get<Patient>(`/patients/${patient_id}`),

  getRecords: (patient_id: string, start_hour?: number, end_hour?: number) =>
    api.get<HourlyRecord[]>(`/patients/${patient_id}/records`, {
      params: { start_hour, end_hour }
    }),

  getStats: () =>
    api.get<{ total_patients: number; sepsis_patients: number; non_sepsis_patients: number; sepsis_rate: number }>('/patients/stats/summary'),
};

export const predictionsApi = {
  create: (data: PredictionRequest) =>
    api.post<PredictionResponse>('/predictions', data),

  getByPatient: (patient_id: string) =>
    api.get<PredictionResponse[]>(`/predictions/patient/${patient_id}`),

  getRecent: (limit = 50) =>
    api.get<PredictionResponse[]>(`/predictions/recent`, { params: { limit } }),
};

export const explanationsApi = {
  getPatient: (patient_id: string, iculos: number) =>
    api.get<ExplanationResponse>(`/explanations/patient/${patient_id}/${iculos}`),

  getGlobalSHAP: () =>
    api.get<Array<{ feature: string; mean_abs_shap: number }>>('/explanations/global/shap'),

  getTemporalSHAP: () =>
    api.get<Array<{ feature: string; 'Early (hour 1-6)': number; 'Mid (hour 7-24)': number; 'Late (hour 25+)': number }>>('/explanations/temporal/shap'),

  getGlobalSHAPImage: () => '/api/explanations/images/global_shap',
  getTemporalSHAPImage: () => '/api/explanations/images/temporal_shap',
  getPatientWaterfallImage: (label: 'TP' | 'TN') => `/api/explanations/images/examples/waterfall/${label}`,
  getPatientLIMEImage: (label: 'TP' | 'TN') => `/api/explanations/images/examples/lime/${label}`,
};

export const metricsApi = {
  getModels: () =>
    api.get<ModelMetrics[]>('/metrics/models'),

  getOverview: () =>
    api.get<OverviewResponse>('/metrics/overview'),

  getExternal: () =>
    api.get<ExternalValidation>('/metrics/external'),
};

export default api;
