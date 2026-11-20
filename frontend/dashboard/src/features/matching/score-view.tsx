import { cn } from "cn";
import { CircleAlertIcon, CircleCheckIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { BoilerplateScore, MatchScore, PairScores } from "@/lib/api";
import { percent } from "@/lib/format";

/** The four requirement questions: yes means the cohort requires it and the patient documentedly meets it. */
const REQUIREMENTS: { key: keyof MatchScore; label: string }[] = [
  { key: "cancer_type", label: "Cancer type" },
  { key: "burden", label: "Disease extent" },
  { key: "treatment_history", label: "Treatment history" },
  { key: "biomarker", label: "Biomarker" },
];

/**
 * Grades from P(reasonable), the decision model's potentially-eligible answer (user, 2026-11-17): Possible at
 * 0.5 or above (the inference package's match_quality_pass), Equivocal from 0.2, else Unlikely. Computed here from
 * the stored probability, so cached scores need no recompute. match_quality_score still ranks the lists.
 */
export function matchLabel(match: MatchScore): { text: string; tone: "good" | "fair" | "poor" } {
  if (match.reasonableness >= 0.5) return { text: "Possible match", tone: "good" };
  if (match.reasonableness >= 0.2) return { text: "Equivocal match", tone: "fair" };
  return { text: "Unlikely match", tone: "poor" };
}

const TONES = {
  good: "bg-green-100 text-green-800",
  fair: "bg-amber-100 text-amber-800",
  poor: "bg-gray-100 text-gray-600",
};

/** The match score and its label, or a spinner while it's being scored. */
export function MatchBadge({
  scores,
  scoring,
}: {
  scores: PairScores | undefined;
  scoring: boolean;
}) {
  const match = scores?.match;
  if (!match) {
    return scoring ? (
      <span className="flex items-center gap-1.5 text-xs text-gray-500">
        <Spinner className="size-3.5" /> Scoring…
      </span>
    ) : (
      <span className="text-xs text-gray-400">Not scored</span>
    );
  }
  const label = matchLabel(match);
  return (
    <Tooltip>
      <TooltipTrigger render={<span className="inline-flex" />}>
        <Badge className={cn("h-6 gap-1.5 px-2.5 text-xs", TONES[label.tone])}>
          {label.text}
          <span className="font-semibold tabular-nums">{match.score.toFixed(2)}</span>
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">
        MatchMiner-AI decision model: probability-weighted match score (0–1). Reasonable
        consideration: {percent(match.reasonableness)}.
      </TooltipContent>
    </Tooltip>
  );
}

/** The requirements the model found the cohort to have and the patient to meet: decision-model match signals,
 * not frank eligibility requirements, hence "Matches based on" (user, 2026-11-12). */
export function Requirements({ match }: { match: MatchScore | null | undefined }) {
  if (!match) return null;
  const met = REQUIREMENTS.filter((r) => (match[r.key] as number) >= 0.5);
  if (!met.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span className="text-gray-500">Matches based on:</span>
      {met.map((r) => (
        <span
          key={r.key}
          className="inline-flex items-center gap-1 rounded bg-green-50 px-1.5 py-0.5 text-green-800"
        >
          <CircleCheckIcon className="size-3" />
          {r.label}
        </span>
      ))}
    </div>
  );
}

/** mm-dash's exclusion line, from the decision model's common-exclusion question. */
export function ExclusionLine({
  boilerplate,
  scoring,
}: {
  boilerplate: BoilerplateScore | null | undefined;
  scoring: boolean;
}) {
  if (!boilerplate) {
    return scoring ? (
      <p className="flex items-center gap-1.5 text-sm text-gray-500">
        <Spinner className="size-3.5" /> Checking common exclusions…
      </p>
    ) : null;
  }
  if ("skipped" in boilerplate) return null;
  return boilerplate.pass ? (
    <p className="flex items-center gap-1.5 text-sm text-green-700">
      <CircleCheckIcon className="size-4 shrink-0" />
      No evidence that the patient meets common exclusion criteria.
    </p>
  ) : (
    <p className="flex items-center gap-1.5 text-sm text-amber-700">
      <CircleAlertIcon className="size-4 shrink-0" />
      The patient may meet one or more common exclusion criteria.
    </p>
  );
}

export function likelyExcluded(scores: PairScores | undefined): boolean {
  const b = scores?.boilerplate;
  return Boolean(b && !("skipped" in b) && !b.pass);
}

/** Scored rows first by match score, then the rest by similarity. */
export function byScore<T extends { id: string; cosine: number }>(
  rows: T[],
  scores: Record<string, PairScores>,
): T[] {
  return [...rows].sort((a, b) => {
    const sa = scores[a.id]?.match?.score;
    const sb = scores[b.id]?.match?.score;
    if (sa !== undefined && sb !== undefined) return sb - sa;
    if (sa !== undefined) return -1;
    if (sb !== undefined) return 1;
    return b.cosine - a.cosine;
  });
}
