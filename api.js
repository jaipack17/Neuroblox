// Frontend API — polling pattern per spec.
// Base URL: http://localhost:8000 (or your deployed IP)

const API_BASE = 'http://localhost:8000';

async function apiStartTraining(payload) {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = res.status === 422 && data.detail
      ? 'Validation: ' + JSON.stringify(data.detail)
      : data.message || data.detail || `HTTP ${res.status}`;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return data; // { message, job_id, monitor_url }
}

async function apiGetStatus(jobId) {
  const res = await fetch(`${API_BASE}/status/${jobId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiGetModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiPredict(jobId, inputData) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, input_data: inputData })
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
