import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { UsersIcon } from "lucide-react";

import { CriteriaList } from "@/components/criteria-list";
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
import { VERSION_LABELS } from "@/lib/api";
import { isoWhen } from "@/lib/format";
import { cohortOptions } from "@/lib/queries";

/** A cohort's complete criteria, opened with `?cohort=<id>`. */
export function CohortDialog({
  cohortId,
  onClose,
  onDetails,
}: {
  cohortId: string;
  onClose: () => void;
  onDetails: (trialId: string) => void;
}) {
  const cohort = useQuery(cohortOptions(cohortId));
  const c = cohort.data;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-3xl" bodyClassName="gap-4">
        {c ? (
          <>
            <DialogHeader>
              <DialogTitle>
                {c.trial.protocol_no || c.trial.nct_id}: {c.name}
              </DialogTitle>
              <DialogDescription className="text-gray-600">
                {c.trial.short_title} · {VERSION_LABELS[c.version]} criteria
                {c.completed_at ? `, extracted ${isoWhen(c.completed_at)}` : ""}
              </DialogDescription>
            </DialogHeader>
            <CriteriaList items={c.items} />
          </>
        ) : cohort.isError ? (
          <DialogTitle>Could not load the cohort: {cohort.error.message}</DialogTitle>
        ) : (
          <>
            <DialogTitle className="sr-only">Loading</DialogTitle>
            <div className="flex justify-center py-12">
              <Spinner className="size-6" />
            </div>
          </>
        )}
        <DialogFooter className="pt-1">
          {c ? (
            <>
              <Button
                variant="outline"
                onClick={() => {
                  onDetails(c.trial.id);
                }}
              >
                Trial details
              </Button>
              <Button
                render={
                  <Link to="/cohorts/$cohortId/patients" params={{ cohortId: c.cohort_id }} />
                }
              >
                <UsersIcon />
                Find patients
              </Button>
            </>
          ) : null}
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
