/**
 * Thin client around the PPE Detection API.
 *
 * Every call returns a plain object; errors are normalised into an `ApiError`
 * that carries the HTTP status and the machine readable error code returned
 * by the backend (`invalid_image`, `model_unavailable`, ...).
 */

const RAW_BASE = import.meta.env?.VITE_API_URL ?? 'http://localhost:8000';

/** The backend is reachable under the same origin in production (nginx proxy). */
export const API_BASE = RAW_BASE.replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status, code) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

async function readError(response) {
  let detail = `Request failed with status ${response.status}`;
  let code = 'error';
  try {
    const body = await response.json();
    if (body?.detail) detail = body.detail;
    if (body?.error) code = body.error;
  } catch {
    /* the body was not JSON - keep the default message */
  }
  return new ApiError(detail, response.status, code);
}

/** GET /health */
export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw await readError(response);
  return response.json();
}

/** GET /model/metadata */
export async function fetchMetadata() {
  const response = await fetch(`${API_BASE}/model/metadata`);
  if (!response.ok) throw await readError(response);
  return response.json();
}

/** GET /samples */
export async function fetchSamples() {
  const response = await fetch(`${API_BASE}/samples`);
  if (!response.ok) throw await readError(response);
  return response.json();
}

/**
 * POST /predict
 * @param {Blob|File} file the image to analyse
 * @param {{conf?: number, iou?: number}} options
 */
export async function predict(file, options = {}) {
  const form = new FormData();
  form.append('file', file, file.name ?? 'frame.jpg');

  const params = new URLSearchParams();
  if (typeof options.conf === 'number') params.set('conf', String(options.conf));
  if (typeof options.iou === 'number') params.set('iou', String(options.iou));
  const query = params.toString();

  const response = await fetch(`${API_BASE}/predict${query ? `?${query}` : ''}`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) throw await readError(response);
  return response.json();
}

/**
 * POST /predict/visualize - returns a Blob of the annotated PNG.
 */
export async function predictVisualize(file, options = {}) {
  const form = new FormData();
  form.append('file', file, file.name ?? 'frame.jpg');

  const params = new URLSearchParams();
  if (typeof options.conf === 'number') params.set('conf', String(options.conf));
  if (typeof options.iou === 'number') params.set('iou', String(options.iou));
  const query = params.toString();

  const response = await fetch(`${API_BASE}/predict/visualize${query ? `?${query}` : ''}`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) throw await readError(response);
  return response.blob();
}

/** Absolute URL of a sample image. */
export function sampleUrl(name) {
  return `${API_BASE}/samples/${encodeURIComponent(name)}`;
}
