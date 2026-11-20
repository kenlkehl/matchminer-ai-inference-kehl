import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { type CohortHit, type PairScores, VERSION_LABELS } from "@/lib/api";
import { cohortOptions } from "@/lib/queries";

import { CriteriaCheck } from "../matching/criteria-check";
import { ExclusionLine, MatchBadge, Requirements } from "../matching/score-view";

/** A matched trial's cohort: its facts, the decision model's answers and the per-criterion check. */
export function TrialDialog({
  row,
  patientId,
  scores,
  onClose,
}: {
  row: CohortHit;
  patientId: string;
  scores: PairScores | undefined;
  onClose: () => void;
}) {
  const cohort = useQuery(cohortOptions(row.cohort.cohort_id));
  const t = row.cohort.trial;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-3xl" bodyClassName="gap-4">
        <DialogHeader>
          <DialogTitle>
            {t.protocol_no || t.nct_id}: {t.short_title}
          </DialogTitle>
          <DialogDescription className="text-gray-600">
            {row.cohort.name} · {t.phase} · {t.center} · {VERSION_LABELS[row.cohort.version]}{" "}
            criteria
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          <MatchBadge scores={scores} scoring={false} />
          <Requirements match={scores?.match} />
          <ExclusionLine boilerplate={scores?.boilerplate} scoring={false} />
        </div>
        {cohort.data ? (
          <CriteriaCheck
            patientId={patientId}
            cohortId={row.cohort.cohort_id}
            items={cohort.data.items}
          />
        ) : (
          <Spinner className="size-5" />
        )}
        <DialogFooter className="pt-1">
          <Button
            variant="outline"
            render={<Link to="/" search={{ center: t.center_id, details: t.id }} />}
          >
            Open on the dashboard
          </Button>
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
