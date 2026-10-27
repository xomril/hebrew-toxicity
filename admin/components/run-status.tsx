import type { PublicationStatus, RunStatus } from "@/lib/api";
import { Badge } from "./ui";

const runLabels: Record<RunStatus, [string, "neutral" | "accent" | "warn" | "safe" | "toxic"]> = {
  queued: ["בתור", "neutral"],
  running: ["רץ", "accent"],
  canceling: ["מבטל…", "warn"],
  succeeded: ["הושלם", "safe"],
  failed: ["נכשל", "toxic"],
  canceled: ["בוטל", "neutral"],
};

export function RunStatusBadge({ status }: { status: RunStatus }) {
  const [label, tone] = runLabels[status];
  return <Badge tone={tone}>{label}</Badge>;
}

const pubLabels: Record<PublicationStatus, [string, "neutral" | "accent" | "safe" | "toxic"]> = {
  queued: ["בתור", "neutral"],
  uploading: ["מעלה…", "accent"],
  published: ["פורסם", "safe"],
  failed: ["נכשל", "toxic"],
};

export function PublicationStatusBadge({ status }: { status: PublicationStatus }) {
  const [label, tone] = pubLabels[status];
  return <Badge tone={tone}>{label}</Badge>;
}
