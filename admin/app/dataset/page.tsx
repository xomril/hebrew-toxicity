"use client";

import { useState, type FormEvent } from "react";
import useSWR from "swr";
import { Button, Card, Empty, ErrorNote, inputClass, LabelBadge, PageHeader, Stat, Badge } from "@/components/ui";
import { api, fetcher, type DatasetStats, type Example, type Label, type Page } from "@/lib/api";
import { when } from "@/lib/format";

const PAGE_SIZE = 25;

const sourceLabels: Record<string, string> = { seed: "ייבוא", manual: "ידני", feedback: "משוב" };

export default function DatasetPage() {
  const [label, setLabel] = useState<"" | "0" | "1">("");
  const [source, setSource] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);

  const params = new URLSearchParams({ limit: String(PAGE_SIZE), offset: String(page * PAGE_SIZE) });
  if (label) params.set("label", label);
  if (source) params.set("source", source);
  if (search.trim()) params.set("search", search.trim());

  const list = useSWR<Page<Example>>(`/dataset?${params}`, fetcher, { keepPreviousData: true });
  const stats = useSWR<DatasetStats>("/dataset/stats", fetcher);
  const [actionError, setActionError] = useState<unknown>(null);

  const refresh = () => Promise.all([list.mutate(), stats.mutate()]);

  async function run(action: () => Promise<unknown>) {
    setActionError(null);
    try {
      await action();
      await refresh();
    } catch (e) {
      setActionError(e);
    }
  }

  const totalPages = list.data ? Math.max(1, Math.ceil(list.data.total / PAGE_SIZE)) : 1;
  const resetPage = <T,>(set: (v: T) => void) => (v: T) => {
    set(v);
    setPage(0);
  };

  return (
    <>
      <PageHeader title="דאטה-סט" description="הדוגמאות המתויגות שעליהן המודל מתאמן" />

      <div className="mb-6 grid grid-cols-3 gap-3">
        <Stat label="סה״כ" value={stats.data?.total ?? "—"} />
        <Stat label="רעילות" value={<span className="text-toxic">{stats.data?.toxic ?? "—"}</span>} />
        <Stat label="תקינות" value={<span className="text-safe">{stats.data?.safe ?? "—"}</span>} />
      </div>

      <AddExample onAdd={(text, l) => run(() => api.post("/dataset", { text, label: l }))} />

      <Card className="mt-6">
        <div className="mb-4 grid gap-2 sm:grid-cols-[1fr_auto_auto]">
          <input className={inputClass} placeholder="חיפוש טקסט…" value={search} onChange={(e) => resetPage(setSearch)(e.target.value)} />
          <select className={inputClass} value={label} onChange={(e) => resetPage(setLabel)(e.target.value as "" | "0" | "1")} aria-label="סינון לפי תגית">
            <option value="">כל התגיות</option>
            <option value="1">רעיל</option>
            <option value="0">תקין</option>
          </select>
          <select className={inputClass} value={source} onChange={(e) => resetPage(setSource)(e.target.value)} aria-label="סינון לפי מקור">
            <option value="">כל המקורות</option>
            <option value="seed">ייבוא</option>
            <option value="manual">ידני</option>
            <option value="feedback">משוב</option>
          </select>
        </div>

        <ErrorNote error={list.error ?? actionError} />

        {list.data && list.data.items.length === 0 ? (
          <Empty>אין דוגמאות שמתאימות לסינון</Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-muted">
                <tr className="border-b border-line text-right">
                  <th className="py-2 font-medium">טקסט</th>
                  <th className="py-2 font-medium">תגית</th>
                  <th className="hidden py-2 font-medium sm:table-cell">מקור</th>
                  <th className="hidden py-2 font-medium md:table-cell">נוסף</th>
                  <th className="py-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {list.data?.items.map((ex) => (
                  <tr key={ex.id}>
                    <td className="max-w-md py-2 pl-3">{ex.text}</td>
                    <td className="py-2">
                      <button
                        title="החלף תגית"
                        onClick={() => run(() => api.patch(`/dataset/${ex.id}`, { label: ex.label === 1 ? 0 : 1 }))}
                        className="rounded-full hover:ring-2 hover:ring-accent/30"
                      >
                        <LabelBadge label={ex.label} />
                      </button>
                    </td>
                    <td className="hidden py-2 sm:table-cell">
                      <Badge>{sourceLabels[ex.source] ?? ex.source}</Badge>
                    </td>
                    <td className="hidden py-2 text-muted md:table-cell">{when(ex.created_at)}</td>
                    <td className="py-2 text-left">
                      <Button
                        variant="ghost"
                        className="h-7 px-2 text-xs"
                        onClick={() => confirm("למחוק את הדוגמה?") && run(() => api.delete(`/dataset/${ex.id}`))}
                      >
                        מחיקה
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-4 flex items-center justify-between text-sm text-muted">
          <span>{list.data ? `${list.data.total} דוגמאות` : "טוען…"}</span>
          <div className="flex items-center gap-2">
            <Button variant="ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>
              הקודם
            </Button>
            <span className="tabular-nums">
              {page + 1} / {totalPages}
            </span>
            <Button variant="ghost" disabled={page + 1 >= totalPages} onClick={() => setPage(page + 1)}>
              הבא
            </Button>
          </div>
        </div>
      </Card>
    </>
  );
}

function AddExample({ onAdd }: { onAdd: (text: string, label: Label) => Promise<void> }) {
  const [text, setText] = useState("");
  const [label, setLabel] = useState<Label>(1);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    await onAdd(text.trim(), label);
    setBusy(false);
    setText("");
  }

  return (
    <Card title="הוספת דוגמה">
      <form onSubmit={submit} className="flex flex-col gap-2 sm:flex-row">
        <input className={inputClass} placeholder="טקסט התגובה…" value={text} onChange={(e) => setText(e.target.value)} />
        <div className="flex shrink-0 gap-2">
          <select className={inputClass} value={label} onChange={(e) => setLabel(Number(e.target.value) as Label)} aria-label="תגית">
            <option value={1}>רעיל</option>
            <option value={0}>תקין</option>
          </select>
          <Button variant="primary" type="submit" disabled={busy || !text.trim()}>
            הוספה
          </Button>
        </div>
      </form>
    </Card>
  );
}
