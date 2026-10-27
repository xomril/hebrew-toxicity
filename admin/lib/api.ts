export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, `השרת לא זמין (${API_URL})`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : Array.isArray(detail) ? detail[0]?.msg : res.statusText;
    throw new ApiError(res.status, message ?? `HTTP ${res.status}`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const fetcher = <T,>(path: string) => request<T>(path);

export const api = {
  get: <T,>(path: string) => request<T>(path),
  post: <T,>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
  patch: <T,>(path: string, body: unknown) => request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  delete: (path: string) => request<void>(path, { method: "DELETE" }),
};

// ---------- types (mirror api/schemas.py) ----------
export type Label = 0 | 1;

export interface Example {
  id: number;
  text: string;
  label: Label;
  source: string;
  raw_comment_id: number | null;
  created_at: string;
}

export interface Page<T> {
  total: number;
  items: T[];
}

export interface DatasetStats {
  total: number;
  safe: number;
  toxic: number;
}

export interface Comment {
  id: number;
  text: string;
  source: string | null;
  predicted_label: Label | null;
  toxic_prob: number | null;
  model_version: string | null;
  created_at: string;
}

export interface Classification {
  comment_id: number;
  text: string;
  is_toxic: boolean;
  label: Label;
  toxic_prob: number;
  model_version: string;
}

export type RunStatus = "queued" | "running" | "canceling" | "succeeded" | "failed" | "canceled";

export interface EpochMetrics {
  epoch: number;
  accuracy: number | null;
  f1: number | null;
  loss: number | null;
}

export interface TrainingRun {
  id: number;
  version: string;
  status: RunStatus;
  params: { epochs: number; learning_rate: number; batch_size: number };
  progress: number;
  current_epoch: number | null;
  history: EpochMetrics[];
  accuracy: number | null;
  f1: number | null;
  dataset_size: number | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export type PublicationStatus = "queued" | "uploading" | "published" | "failed";

export interface Publication {
  id: number;
  version: string;
  repo_id: string;
  private: boolean;
  status: PublicationStatus;
  url: string | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface ModelVersion {
  version: string;
  is_default: boolean;
  run: TrainingRun | null;
  publications: Publication[];
}

export interface HubStatus {
  logged_in: boolean;
  user: string | null;
  default_repo_id: string | null;
}

export const ACTIVE_RUN_STATUSES: RunStatus[] = ["queued", "running", "canceling"];
