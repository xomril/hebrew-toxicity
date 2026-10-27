"use client";

import Link from "next/link";
import useSWR from "swr";
import { RunStatusBadge } from "@/components/run-status";
import { Card, Empty, ErrorNote, PageHeader, ProgressBar, Stat } from "@/components/ui";
import { ACTIVE_RUN_STATUSES, fetcher, type DatasetStats, type HubStatus, type ModelVersion, type Page, type Comment, type TrainingRun } from "@/lib/api";
import { pct, when } from "@/lib/format";

export default function OverviewPage() {
  const stats = useSWR<DatasetStats>("/dataset/stats", fetcher);
  const models = useSWR<ModelVersion[]>("/models", fetcher);
  const unlabeled = useSWR<Page<Comment>>("/comments?unlabeled=true&limit=1", fetcher);
  const runs = useSWR<TrainingRun[]>("/training/runs?limit=5", fetcher, {
    refreshInterval: (data) => (data?.some((r) => ACTIVE_RUN_STATUSES.includes(r.status)) ? 2000 : 0),
  });
  const hub = useSWR<HubStatus>("/models/hub", fetcher);

  const current = models.data?.find((m) => m.is_default);
  const activeRun = runs.data?.find((r) => ACTIVE_RUN_STATUSES.includes(r.status));

  return (
    <>
      <PageHeader title="סקירה" description="מצב הדאטה, המודלים והאימונים" />
      <ErrorNote error={stats.error ?? models.error} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="דוגמאות מתויגות" value={stats.data?.total ?? "—"} hint={stats.data && `${stats.data.toxic} רעילות · ${stats.data.safe} תקינות`} />
        <Stat label="ממתינות לתיוג" value={unlabeled.data?.total ?? "—"} hint={<Link href="/review" className="text-accent hover:underline">לתור התיוג</Link>} />
        <Stat label="מודל פעיל" value={<span dir="ltr" className="text-base">{current?.version ?? "—"}</span>} hint={current?.run ? `דיוק ${pct(current.run.accuracy)} · F1 ${pct(current.run.f1)}` : "אומן לפני מעקב ריצות"} />
        <Stat
          label="Hugging Face"
          value={<span className="text-base">{hub.data ? (hub.data.logged_in ? "מחובר" : "לא מחובר") : "—"}</span>}
          hint={hub.data?.user ? <span dir="ltr">{hub.data.user}</span> : hub.data && <Link href="/models" className="text-accent hover:underline">איך מתחברים</Link>}
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="אימון פעיל" action={<Link href="/training" className="text-sm text-accent hover:underline">לכל האימונים</Link>}>
          {activeRun ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span dir="ltr" className="font-medium">{activeRun.version}</span>
                <RunStatusBadge status={activeRun.status} />
              </div>
              <ProgressBar value={activeRun.progress} />
              <div className="text-sm text-muted">
                {Math.round(activeRun.progress * 100)}% · אפוק {activeRun.current_epoch?.toFixed(1) ?? 0} מתוך {activeRun.params.epochs}
              </div>
            </div>
          ) : (
            <Empty>אין אימון רץ כרגע</Empty>
          )}
        </Card>

        <Card title="אימונים אחרונים">
          {runs.data?.length ? (
            <ul className="divide-y divide-line">
              {runs.data.map((r) => (
                <li key={r.id} className="flex items-center justify-between gap-3 py-2 text-sm">
                  <span dir="ltr" className="truncate">{r.version}</span>
                  <span className="flex items-center gap-3">
                    <span className="tabular-nums text-muted">{pct(r.accuracy)}</span>
                    <RunStatusBadge status={r.status} />
                    <span className="hidden text-muted sm:inline">{when(r.created_at)}</span>
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>עוד לא הורצו אימונים מהממשק</Empty>
          )}
        </Card>
      </div>
    </>
  );
}
