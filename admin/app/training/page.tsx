"use client";

import { useState, type FormEvent } from "react";
import useSWR from "swr";
import { RunStatusBadge } from "@/components/run-status";
import { Button, Card, Empty, ErrorNote, inputClass, PageHeader, ProgressBar } from "@/components/ui";
import { ACTIVE_RUN_STATUSES, api, fetcher, type DatasetStats, type TrainingRun } from "@/lib/api";
import { duration, pct, when } from "@/lib/format";

export default function TrainingPage() {
  const runs = useSWR<TrainingRun[]>("/training/runs?limit=50", fetcher, {
    // רענון חי רק כשיש אימון פעיל
    refreshInterval: (data) => (data?.some((r) => ACTIVE_RUN_STATUSES.includes(r.status)) ? 2000 : 0),
  });

  return (
    <>
      <PageHeader title="אימון" description="הרצת אימון חדש על הדאטה-סט הנוכחי ומעקב אחרי התקדמות" />
      <NewRunForm onCreated={() => runs.mutate()} />
      <Card title="ריצות" className="mt-6">
        <ErrorNote error={runs.error} />
        {runs.data?.length === 0 && <Empty>עוד לא הורצו אימונים</Empty>}
        <ul className="divide-y divide-line">
          {runs.data?.map((run) => (
            <RunRow key={run.id} run={run} onChange={() => runs.mutate()} />
          ))}
        </ul>
      </Card>
    </>
  );
}

function NewRunForm({ onCreated }: { onCreated: () => void }) {
  const stats = useSWR<DatasetStats>("/dataset/stats", fetcher);
  const [version, setVersion] = useState("");
  const [epochs, setEpochs] = useState(5);
  const [lr, setLr] = useState("2e-5");
  const [batch, setBatch] = useState(8);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.post("/training/runs", {
        version: version.trim() || undefined,
        epochs,
        learning_rate: Number(lr),
        batch_size: batch,
      });
      setVersion("");
      onCreated();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="אימון חדש">
      <form onSubmit={submit} className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="שם גרסה" hint="ריק = נוצר אוטומטית לפי תאריך">
            <input dir="ltr" className={inputClass} placeholder="v20261027_1800" value={version} onChange={(e) => setVersion(e.target.value)} />
          </Field>
          <Field label={`אפוקים: ${epochs}`} hint="כמה פעמים לעבור על הדאטה">
            <input type="range" min={1} max={20} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} className="h-9 w-full accent-accent" />
          </Field>
          <Field label="Learning rate">
            <select dir="ltr" className={inputClass} value={lr} onChange={(e) => setLr(e.target.value)}>
              {["1e-5", "2e-5", "3e-5", "5e-5", "1e-4"].map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </Field>
          <Field label="Batch size">
            <select className={inputClass} value={batch} onChange={(e) => setBatch(Number(e.target.value))}>
              {[4, 8, 16, 32].map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </Field>
        </div>
        <ErrorNote error={error} />
        <div className="flex items-center gap-3">
          <Button variant="primary" type="submit" disabled={busy}>
            התחלת אימון
          </Button>
          <span className="text-sm text-muted">יאומן על {stats.data?.total ?? "…"} דוגמאות (80% אימון, 20% בדיקה)</span>
        </div>
      </form>
    </Card>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </label>
  );
}

function RunRow({ run, onChange }: { run: TrainingRun; onChange: () => void }) {
  const [open, setOpen] = useState(false);
  const active = ACTIVE_RUN_STATUSES.includes(run.status);
  const [error, setError] = useState<unknown>(null);

  async function cancel() {
    setError(null);
    try {
      await api.post(`/training/runs/${run.id}/cancel`);
      onChange();
    } catch (e) {
      setError(e);
    }
  }

  return (
    <li className="py-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <RunStatusBadge status={run.status} />
          <span dir="ltr" className="truncate font-medium">{run.version}</span>
        </div>
        <div className="flex items-center gap-4 text-sm">
          {run.status === "succeeded" && (
            <span className="tabular-nums">
              דיוק <b>{pct(run.accuracy)}</b> · F1 <b>{pct(run.f1)}</b>
            </span>
          )}
          <span className="hidden text-muted sm:inline">{when(run.created_at)}</span>
          {(run.status === "queued" || run.status === "running") && (
            <Button variant="danger" className="h-7 px-2 text-xs" onClick={cancel}>
              ביטול
            </Button>
          )}
          {run.history.length > 0 && (
            <Button variant="ghost" className="h-7 px-2 text-xs" onClick={() => setOpen(!open)}>
              {open ? "הסתר" : "פירוט"}
            </Button>
          )}
        </div>
      </div>

      {active && (
        <div className="mt-3 space-y-1">
          <ProgressBar value={run.progress} tone={run.status === "canceling" ? "muted" : "accent"} />
          <div className="flex justify-between text-xs text-muted">
            <span>
              {Math.round(run.progress * 100)}% · אפוק {run.current_epoch?.toFixed(1) ?? 0} מתוך {run.params.epochs}
            </span>
            <span>{run.status === "queued" ? "ממתין ל-worker" : duration(run.started_at, null)}</span>
          </div>
        </div>
      )}

      <div className="mt-2 text-xs text-muted">
        <span dir="ltr">
          epochs={run.params.epochs} · lr={run.params.learning_rate} · batch={run.params.batch_size}
        </span>
        {run.dataset_size != null && <> · {run.dataset_size} דוגמאות</>}
        {run.finished_at && <> · משך {duration(run.started_at, run.finished_at)}</>}
      </div>

      {run.error && <p className="mt-2 text-xs text-toxic" dir="ltr">{run.error}</p>}
      <ErrorNote error={error} />

      {open && (
        <table className="mt-3 w-full max-w-md text-xs">
          <thead className="text-muted">
            <tr className="text-right">
              <th className="py-1 font-medium">אפוק</th>
              <th className="py-1 font-medium">דיוק</th>
              <th className="py-1 font-medium">F1</th>
              <th className="py-1 font-medium">Loss</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {run.history.map((h) => (
              <tr key={h.epoch} className="border-t border-line">
                <td className="py-1">{h.epoch}</td>
                <td className="py-1">{pct(h.accuracy)}</td>
                <td className="py-1">{pct(h.f1)}</td>
                <td className="py-1">{h.loss?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </li>
  );
}
