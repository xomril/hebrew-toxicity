"use client";

import { useState } from "react";
import useSWR from "swr";
import { Button, Card, Empty, ErrorNote, LabelBadge, PageHeader } from "@/components/ui";
import { api, fetcher, type Comment, type Label, type Page } from "@/lib/api";
import { pct, when } from "@/lib/format";

export default function ReviewPage() {
  const queue = useSWR<Page<Comment>>("/comments?unlabeled=true&limit=50", fetcher);
  const [error, setError] = useState<unknown>(null);
  const [pending, setPending] = useState<number | null>(null);

  async function label(comment: Comment, correct: Label) {
    setError(null);
    setPending(comment.id);
    try {
      await api.post("/feedback", { comment_id: comment.id, correct_label: correct });
      await queue.mutate();
    } catch (e) {
      setError(e);
    } finally {
      setPending(null);
    }
  }

  return (
    <>
      <PageHeader
        title="תור תיוג"
        description="תגובות שסווגו דרך ה-API ועוד לא תויגו. כל תיוג נכנס לדאטה-סט לאימון הבא."
      />
      <ErrorNote error={queue.error ?? error} />

      <Card title={queue.data ? `${queue.data.total} תגובות ממתינות` : undefined}>
        {queue.data?.items.length === 0 ? (
          <Empty>אין תגובות לתיוג. תגובות שנשלחות ל-POST /classify יופיעו כאן.</Empty>
        ) : (
          <ul className="divide-y divide-line">
            {queue.data?.items.map((c) => (
              <li key={c.id} className="flex flex-col gap-3 py-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <p>{c.text}</p>
                  <p className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted">
                    המודל אמר: <LabelBadge label={c.predicted_label} /> ({pct(c.toxic_prob)} רעילות) · {when(c.created_at)}
                    {c.source && <> · {c.source}</>}
                  </p>
                </div>
                <div className="flex shrink-0 gap-2">
                  <Button variant="danger" disabled={pending === c.id} onClick={() => label(c, 1)}>
                    רעיל
                  </Button>
                  <Button disabled={pending === c.id} onClick={() => label(c, 0)}>
                    תקין
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </>
  );
}
