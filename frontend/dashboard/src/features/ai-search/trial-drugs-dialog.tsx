import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";

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
import type { Meta, TrialSummary } from "@/lib/api";
import { trialOptions } from "@/lib/queries";

import type { OpenEvidence } from "../overview/catalog-links";
import { DrugsSection } from "../overview/drugs-section";

/** Why a trial's drugs can't be shown (the Drugs / mechanism button is then disabled), or null. */
export function drugsUnavailable(trial: TrialSummary, meta: Meta | undefined): string | null {
  if (trial.drugs === undefined) {
    return "This server predates the trials' drug information: restart it to show them.";
  }
  const catalog = meta?.good_options;
  if (catalog === null) {
    return "The dashboard's store was built before the trials' drug information; the server rebuilds it shortly.";
  }
  if (catalog && catalog.state !== "ready") {
    return catalog.message ?? "The GoodOption catalog could not be read.";
  }
  if (trial.drugs.length === 0) {
    return "The GoodOption catalog lists no drugs for this trial (supportive care aside).";
  }
  return null;
}

/**
 * A trial's drugs and mechanisms from the GoodOption catalog (`?drugs=<trialId>` on the
 * trials-for-a-patient view): trial details' Drugs section, grouped as studied, "vs" and "with".
 * `children` (the catalog evidence dialog) render inside it, so they stack over it.
 */
export function TrialDrugsDialog({
  trialId,
  onClose,
  onEvidence,
  children,
}: {
  trialId: string;
  onClose: () => void;
  onEvidence: OpenEvidence;
  children?: ReactNode;
}) {
  const trial = useQuery(trialOptions(trialId));
  const t = trial.data;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-2xl" bodyClassName="gap-4">
        {t ? (
          <>
            <DialogHeader>
              <DialogTitle>Drugs / mechanism: {t.protocol_no || t.nct_id}</DialogTitle>
              <DialogDescription className="text-gray-600">
                {[t.short_title, t.phase, t.center].filter(Boolean).join(" · ")}
              </DialogDescription>
            </DialogHeader>
            {t.drugs?.length ? (
              <DrugsSection drugs={t.drugs} onEvidence={onEvidence} grouped heading={false} />
            ) : (
              <p className="text-sm text-gray-500">
                The GoodOption catalog lists no drugs for this trial.
              </p>
            )}
          </>
        ) : trial.isError ? (
          <DialogTitle>Could not load the trial: {trial.error.message}</DialogTitle>
        ) : (
          <>
            <DialogTitle className="sr-only">Loading the trial's drugs</DialogTitle>
            <div className="flex justify-center py-12">
              <Spinner className="size-6" />
            </div>
          </>
        )}
        <DialogFooter className="pt-1">
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
        {children}
      </DialogContent>
    </Dialog>
  );
}
