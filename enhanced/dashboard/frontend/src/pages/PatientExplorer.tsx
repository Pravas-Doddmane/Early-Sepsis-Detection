import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Search, ChevronLeft, ChevronRight, Eye, TrendingUp, Users } from 'lucide-react';
import { patientsApi, predictionsApi, Patient, HourlyRecord, PredictionResponse } from '../api';

export default function PatientExplorer() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [search, setSearch] = useState('');
  const [sepsisOnly, setSepsisOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [patientRecords, setPatientRecords] = useState<HourlyRecord[]>([]);
  const [patientPredictions, setPatientPredictions] = useState<PredictionResponse[]>([]);
  const [recordsLoading, setRecordsLoading] = useState(false);

  useEffect(() => {
    loadPatients();
  }, [page, search, sepsisOnly]);

  const loadPatients = async () => {
    setLoading(true);
    try {
      const res = await patientsApi.list({ page, page_size: pageSize, sepsis_only: sepsisOnly });
      setPatients(res.data.patients);
      setTotal(res.data.total);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handlePatientClick = async (patient: Patient) => {
    setSelectedPatient(patient);
    setRecordsLoading(true);
    try {
      const [recordsRes, predsRes] = await Promise.all([
        patientsApi.getRecords(patient.patient_id),
        predictionsApi.getByPatient(patient.patient_id),
      ]);
      setPatientRecords(recordsRes.data);
      setPatientPredictions(predsRes.data);
    } catch (err) {
      console.error(err);
    } finally {
      setRecordsLoading(false);
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px' }}>Patient Explorer</h1>
        <p style={{ color: 'var(--color-text-muted)' }}>
          Browse {total.toLocaleString()} patients from the PhysioNet 2019 Sepsis dataset.
        </p>
      </div>

      <div className="patient-explorer-layout">
        {/* Patient List */}
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <div className="card-title">Patients ({total.toLocaleString()} total)</div>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={sepsisOnly}
                  onChange={(e) => setSepsisOnly(e.target.checked)}
                />
                Sepsis only
              </label>
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <div style={{ position: 'relative', maxWidth: '300px' }}>
              <Search style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--color-text-muted)', width: 18, height: 18 }} />
              <input
                type="text"
                placeholder="Search patient ID..."
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                className="form-input"
                style={{ paddingLeft: '40px' }}
              />
            </div>
          </div>

          <div className="table-container" style={{ flex: 1, overflow: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Age</th>
                  <th>Gender</th>
                  <th>Unit</th>
                  <th>Sepsis</th>
                  <th>Onset</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={7} style={{ textAlign: 'center', padding: '40px' }}>Loading...</td></tr>
                ) : patients.length === 0 ? (
                  <tr><td colSpan={7} style={{ textAlign: 'center', padding: '40px', color: 'var(--color-text-muted)' }}>No patients found</td></tr>
                ) : (
                  patients.map(patient => (
                    <tr
                      key={patient.patient_id}
                      onClick={() => handlePatientClick(patient)}
                      style={{
                        cursor: 'pointer',
                        background: selectedPatient?.patient_id === patient.patient_id
                          ? 'rgba(37, 99, 235, 0.05)'
                          : undefined,
                      }}
                    >
                      <td style={{ fontFamily: 'monospace', fontWeight: 500 }}>{patient.patient_id}</td>
                      <td>{patient.age?.toFixed(0) ?? '—'}</td>
                      <td>{patient.gender === 1 ? 'Male' : patient.gender === 0 ? 'Female' : '—'}</td>
                      <td>{patient.unit1 ?? '—'} / {patient.unit2 ?? '—'}</td>
                      <td>
                        <span className={`badge ${patient.sepsis_label === 1 ? 'badge-danger' : 'badge-success'}`}>
                          {patient.sepsis_label === 1 ? 'Sepsis' : 'No Sepsis'}
                        </span>
                      </td>
                      <td>{patient.onset_hour ? `${patient.onset_hour}h` : '—'}</td>
                      <td style={{ textAlign: 'right' }}>
                        <button
                          type="button"
                          className="btn btn-secondary btn-compact"
                          aria-label={`View details for ${patient.patient_id}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            handlePatientClick(patient);
                          }}
                        >
                          <Eye size={16} /> View details
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '16px', marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--color-border)' }}>
            <button
              className="btn btn-secondary"
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              <ChevronLeft size={16} /> Prev
            </button>
            <span style={{ color: 'var(--color-text-muted)' }}>
              Page {page} of {totalPages}
            </span>
            <button
              className="btn btn-secondary"
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>

        {/* Patient Detail Panel */}
        <div className="card" style={{ height: 'fit-content', position: 'sticky', top: '100px' }}>
          {selectedPatient ? (
            <>
              <div className="card-header">
                <div className="card-title">
                  <span style={{ fontFamily: 'monospace' }}>{selectedPatient.patient_id}</span>
                  <span className={`badge ${selectedPatient.sepsis_label === 1 ? 'badge-danger' : 'badge-success'}`} style={{ marginLeft: '8px' }}>
                    {selectedPatient.sepsis_label === 1 ? 'Sepsis' : 'No Sepsis'}
                  </span>
                </div>
              </div>

              <div style={{ marginBottom: '16px', fontSize: '14px', color: 'var(--color-text-muted)' }}>
                <div>Age: {selectedPatient.age?.toFixed(0) ?? '—'}</div>
                <div>Gender: {selectedPatient.gender === 1 ? 'Male' : selectedPatient.gender === 0 ? 'Female' : '—'}</div>
                <div>Unit: {selectedPatient.unit1 ?? '—'} / {selectedPatient.unit2 ?? '—'}</div>
                <div>Hospital Admission Time: {selectedPatient.hosp_adm_time?.toFixed(1) ?? '—'}h</div>
                {selectedPatient.onset_hour && (
                  <div style={{ color: 'var(--color-danger)' }}>
                    Sepsis Onset: {selectedPatient.onset_hour}h
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                <Link to={`/explain/${selectedPatient.patient_id}`} className="btn btn-primary" style={{ flex: 1 }}>
                  <Eye size={16} /> View Explainability
                </Link>
              </div>

              <div className="card-header" style={{ marginTop: '8px' }}>
                <div className="card-title">Hourly Timeline</div>
              </div>

              <div style={{ maxHeight: '400px', overflow: 'auto' }}>
                <table style={{ fontSize: '12px' }}>
                  <thead>
                    <tr>
                      <th>Hour</th>
                      <th>HR</th>
                      <th>MAP</th>
                      <th>Temp</th>
                      <th>Resp</th>
                      <th>O2Sat</th>
                      <th>Sepsis</th>
                      <th>Pred</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recordsLoading ? (
                      <tr><td colSpan={8} style={{ textAlign: 'center', padding: '20px' }}>Loading...</td></tr>
                    ) : patientRecords.length === 0 ? (
                      <tr><td colSpan={8} style={{ textAlign: 'center', padding: '20px', color: 'var(--color-text-muted)' }}>No records</td></tr>
                    ) : (
                      patientRecords.map(record => {
                        const pred = patientPredictions.find(p => p.iculos === record.iculos);
                        return (
                          <tr key={record.iculos} style={{ background: record.sepsis_label === 1 ? 'rgba(220, 38, 38, 0.05)' : undefined }}>
                            <td style={{ fontWeight: 500 }}>{record.iculos}h</td>
                            <td>{record.hr?.toFixed(0) ?? '—'}</td>
                            <td>{record.map?.toFixed(0) ?? '—'}</td>
                            <td>{record.temp?.toFixed(1) ?? '—'}</td>
                            <td>{record.resp?.toFixed(0) ?? '—'}</td>
                            <td>{record.o2sat?.toFixed(0) ?? '—'}</td>
                            <td>
                              {record.sepsis_label === 1 && <TrendingUp size={14} style={{ color: 'var(--color-danger)' }} />}
                            </td>
                            <td>
                              {pred && (
                                <span className={`badge ${pred.prediction === 1 ? 'badge-danger' : 'badge-success'}`}>
                                  {(pred.prob_sepsis * 100).toFixed(1)}%
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--color-text-muted)' }}>
              <Users size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
              <p>Click a patient to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
