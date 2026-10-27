"use client";

import { useState, type FormEvent } from "react";
import useSWR from "swr";
import { PublicationStatusBadge } from "@/components/run-status";
import { Badge, Button, Card, Empty, ErrorNote, inputClass, LabelBadge, PageHeader } from "@/components/ui";
import { api, fetcher, type Classification, type HubStatus, type ModelVersion } from "@/lib/api";
import { pct, when } from "@/lib/format";

const UPLOADING = ["queued", "uploading"];

export default function ModelsPage() {
  const models = useSWR<ModelVersion[]>("/models", fetcher, {
    refreshInterval: (data) => (data?.some((m) => m.publications.some((p) => UPLOADING.includes(p.status))) ? 3000 : 0),
  });
  const hub = useSWR<HubStatus>("/models/hub", fetcher);

  return (
    <>
      <PageHeader title="מודלים ופרסום" description="בדיקת גרסאות מודל והעלאה ל-Hugging Face כדי שאחרים יוכלו להשתמש בהן" />
      <HubBanner hub={hub.data} onRecheck={() => hub.mutate()} />
      <ErrorNote error={models.error} />
      {models.data?.length === 0 && <Empty>אין מודלים ברג׳יסטרי. הריצו אימון בעמוד האימון.</Empty>}
      <div className="space-y-4">
        {models.data?.map((m) => (
          <ModelCard key={m.version} model={m} hub={hub.data} onChange={() => models.mutate()} />
        ))}
      </div>
    </>
  );
}

function HubBanner({ hub, onRecheck }: { hub?: HubStatus; onRecheck: () => void }) {
  if (!hub || hub.logged_in) return null;
  return (
    <div className="mb-6 rounded-xl border border-warn/40 bg-warn/10 p-4 text-sm">
      <p className="font-medium">לא מחובר ל-Hugging Face</p>
      <p className="mt-1 text-muted">
        כדי לפרסם מודלים, צרו טוקן עם הרשאת <b>Write</b> ב-
        <a className="text-accent hover:underline" href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
          huggingface.co/settings/tokens
        </a>{" "}
        והריצו בטרמינל:
      </p>
      <pre dir="ltr" className="mt-2 rounded-lg bg-surface px-3 py-2 text-xs">venv/bin/hf auth login</pre>
      <p className="mt-2 text-muted">אחרי ההתחברות הפעילו מחדש את ה-API וה-worker.</p>
      <Button className="mt-3" onClick={onRecheck}>
        בדיקה חוזרת
      </Button>
    </div>
  );
}

function ModelCard({ model, hub, onChange }: { model: ModelVersion; hub?: HubStatus; onChange: () => void }) {
  const run = model.run;
  const latestPub = model.publications[0];

  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span dir="ltr" className="text-lg font-semibold">{model.version}</span>
            {model.is_default && <Badge tone="accent">פעיל ב-API</Badge>}
          </div>
          <p className="mt-1 text-sm text-muted">
            {run ? (
              <>
                דיוק <b className="text-fg">{pct(run.accuracy)}</b> · F1 <b className="text-fg">{pct(run.f1)}</b> · {run.dataset_size} דוגמאות ·{" "}
                {when(run.finished_at)}
              </>
            ) : (
              "אומן לפני שהתחלנו לשמור ריצות - אין מטריקות"
            )}
          </p>
        </div>
      </div>

      <div className="mt-5 grid gap-6 lg:grid-cols-2">
        <TryModel version={model.version} />
        <div>
          <h3 className="mb-2 text-xs font-medium">פרסום ל-Hugging Face</h3>
          {model.publications.length > 0 && (
            <ul className="mb-3 space-y-2 text-sm">
              {model.publications.slice(0, 3).map((p) => (
                <li key={p.id} className="flex flex-wrap items-center gap-2">
                  <PublicationStatusBadge status={p.status} />
                  {p.url ? (
                    <a dir="ltr" href={p.url} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                      {p.repo_id}
                    </a>
                  ) : (
                    <span dir="ltr">{p.repo_id}</span>
                  )}
                  <span className="text-xs text-muted">{p.private ? "פרטי" : "ציבורי"}</span>
                  {p.error && (
                    <span dir="ltr" className="w-full text-xs text-toxic">
                      {p.error}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {latestPub?.status === "published" && <UsageSnippet repoId={latestPub.repo_id} version={model.version} />}
          <PublishForm version={model.version} hub={hub} busy={!!latestPub && UPLOADING.includes(latestPub.status)} onPublished={onChange} />
        </div>
      </div>
    </Card>
  );
}

function TryModel({ version }: { version: string }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<Classification | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.post<Classification>("/classify", { text, model_version: version, source: "admin" }));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h3 className="mb-2 text-xs font-medium">בדיקה</h3>
      <form onSubmit={submit} className="flex gap-2">
        <input className={inputClass} placeholder="כתבו תגובה לבדיקה…" value={text} onChange={(e) => setText(e.target.value)} />
        <Button type="submit" disabled={busy || !text.trim()}>
          סיווג
        </Button>
      </form>
      <ErrorNote error={error} />
      {result && (
        <div className="mt-3 flex items-center gap-2 text-sm">
          <LabelBadge label={result.label} />
          <span className="text-muted">{pct(result.toxic_prob)} רעילות</span>
        </div>
      )}
    </div>
  );
}

function PublishForm({ version, hub, busy, onPublished }: { version: string; hub?: HubStatus; busy: boolean; onPublished: () => void }) {
  const [repoId, setRepoId] = useState("");
  const [isPrivate, setIsPrivate] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [sending, setSending] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setSending(true);
    setError(null);
    try {
      await api.post(`/models/${version}/publish`, { repo_id: repoId.trim() || undefined, private: isPrivate });
      onPublished();
    } catch (err) {
      setError(err);
    } finally {
      setSending(false);
    }
  }

  const disabled = !hub?.logged_in || busy || sending;
  return (
    <form onSubmit={submit} className="space-y-2">
      <div className="flex gap-2">
        <input dir="ltr" className={inputClass} placeholder={hub?.default_repo_id ?? "user/hebrew-toxicity"} value={repoId} onChange={(e) => setRepoId(e.target.value)} />
        <Button variant="primary" type="submit" disabled={disabled}>
          {busy ? "מעלה…" : "פרסום"}
        </Button>
      </div>
      <label className="flex items-center gap-2 text-xs text-muted">
        <input type="checkbox" checked={isPrivate} onChange={(e) => setIsPrivate(e.target.checked)} className="accent-accent" />
        repo פרטי (רק מי שתתנו לו גישה יוכל להוריד)
      </label>
      <ErrorNote error={error} />
    </form>
  );
}

function UsageSnippet({ repoId, version }: { repoId: string; version: string }) {
  const code = `from transformers import pipeline\n\nclassifier = pipeline("text-classification", model="${repoId}", revision="${version}")\nclassifier("איזה יום יפה היום")`;
  return (
    <div className="mb-3">
      <p className="mb-1 text-xs text-muted">שימוש (למי שיש גישה ל-repo):</p>
      <pre dir="ltr" className="overflow-x-auto rounded-lg bg-hover px-3 py-2 text-xs">{code}</pre>
    </div>
  );
}
