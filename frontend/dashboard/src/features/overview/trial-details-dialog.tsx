import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { UsersIcon } from "lucide-react";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  type SiteBranding,
  type TrialDetail,
  type TrialVersion,
  VERSION_LABELS,
  type Version,
} from "@/lib/api";
import { isoWhen } from "@/lib/format";
import { trialOptions } from "@/lib/queries";
import { useSite } from "@/lib/site";

import type { OpenEvidence } from "./catalog-links";
import { DrugsSection } from "./drugs-section";

const SECTION_HEADING = "mb-1 text-sm font-bold";

/**
 * A trial's description, contacts and both versions' cohorts, opened with `?details=<trialId>`.
 * `children` (the catalog evidence dialog) render inside it, so they stack over it as a nested dialog.
 */
export function TrialDetailsDialog({
  trialId,
  onClose,
  onCriteria,
  onEvidence,
  children,
}: {
  trialId: string;
  onClose: () => void;
  onCriteria: (cohortId: string) => void;
  onEvidence?: OpenEvidence;
  children?: ReactNode;
}) {
  const trial = useQuery(trialOptions(trialId));
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-3xl" bodyClassName="gap-4">
        {trial.data ? (
          <TrialDetails trial={trial.data} onCriteria={onCriteria} onEvidence={onEvidence} />
        ) : trial.isError ? (
          <DialogTitle>Could not load the trial: {trial.error.message}</DialogTitle>
        ) : (
          <>
            <DialogTitle className="sr-only">Loading trial details</DialogTitle>
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

function links(trial: TrialDetail, site: SiteBranding) {
  const out: { label: string; href: string }[] = [];
  if (trial.nct_id)
    out.push({
      label: "ClinicalTrials.gov",
      href: `https://clinicaltrials.gov/study/${trial.nct_id}`,
    });
  if (trial.protocol_no) {
    const id = encodeURIComponent(trial.protocol_no);
    for (const link of site.trial_links)
      out.push({ label: link.label, href: link.url.replaceAll("{protocol_no}", id) });
  }
  return out;
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  if (!children) return null;
  return (
    <div className="flex gap-2">
      <dt className="w-32 shrink-0 text-gray-500">{label}</dt>
      <dd className="text-gray-800">{children}</dd>
    </div>
  );
}

const STATE_LABELS: Record<string, string> = {
  complete: "Extracted",
  stale: "Re-extraction in progress (showing the previous result)",
  extracting: "Extraction in progress",
  pending: "Not extracted yet",
  not_treatment: "Screened out: not a cancer treatment trial",
  not_run: "Not extracted",
};

function TrialDetails({
  trial,
  onCriteria,
  onEvidence,
}: {
  trial: TrialDetail;
  onCriteria: (cohortId: string) => void;
  onEvidence: OpenEvidence | undefined;
}) {
  const site = useSite();
  const versions = (["protocol", "ctgov"] as Version[]).filter(
    (v) => trial.versions[v] !== undefined,
  );
  return (
    <>
      <DialogHeader className="gap-3">
        <DialogTitle>
          {trial.protocol_no || trial.nct_id}: {trial.short_title}
        </DialogTitle>
        <DialogDescription className="text-gray-600">
          {trial.official_title || trial.long_title}
        </DialogDescription>
      </DialogHeader>
      <dl className="flex flex-col gap-1 text-sm">
        <Fact label="Disease center">{trial.center}</Fact>
        <Fact label="Phase">{trial.phase}</Fact>
        <Fact label="Status">{trial.status}</Fact>
        <Fact label={site.pi_label}>{trial.site_pi}</Fact>
        <Fact label="Overall PI">{trial.overall_pi !== trial.site_pi ? trial.overall_pi : ""}</Fact>
        <Fact label="Contact">
          {trial.contact.name ? (
            <>
              {trial.contact.name}
              {trial.contact.role ? ` (${trial.contact.role})` : ""}
              {trial.contact.email ? (
                <>
                  {" · "}
                  <a className="link" href={`mailto:${trial.contact.email}`}>
                    {trial.contact.email}
                  </a>
                </>
              ) : null}
            </>
          ) : null}
        </Fact>
        <Fact label="Locations">{trial.sites.join(", ")}</Fact>
        <Fact label="Links">
          <span className="flex flex-wrap gap-x-3">
            {links(trial, site).map((l) => (
              <a key={l.label} className="link" href={l.href} target="_blank" rel="noreferrer">
                {l.label}
              </a>
            ))}
          </span>
        </Fact>
      </dl>
      {trial.brief_summary ? (
        <section>
          <h3 className={SECTION_HEADING}>Summary</h3>
          <p className="text-sm whitespace-pre-wrap text-gray-700">{trial.brief_summary}</p>
        </section>
      ) : null}
      {trial.drugs?.length ? <DrugsSection drugs={trial.drugs} onEvidence={onEvidence} /> : null}
      <section>
        <h3 className={SECTION_HEADING}>Cohorts</h3>
        <Tabs defaultValue={trial.primary_version ?? "protocol"}>
          <TabsList>
            {versions.map((v) => (
              <TabsTrigger key={v} value={v}>
                {VERSION_LABELS[v]} ({trial.versions[v].cohorts.length})
              </TabsTrigger>
            ))}
          </TabsList>
          {versions.map((v) => (
            <TabsContent key={v} value={v} className="pt-2">
              <VersionCohorts
                version={trial.versions[v]}
                shown={trial.primary_version === v}
                onCriteria={onCriteria}
              />
            </TabsContent>
          ))}
        </Tabs>
      </section>
    </>
  );
}

function VersionCohorts({
  version,
  shown,
  onCriteria,
}: {
  version: TrialVersion;
  shown: boolean;
  onCriteria: (cohortId: string) => void;
}) {
  const state = STATE_LABELS[version.state] ?? version.state;
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-gray-500">
        {version.excluded_reason ? `Not extracted: ${version.excluded_reason}` : state}
        {version.completed_at ? ` ${isoWhen(version.completed_at)}` : ""}
        {version.extracted_cohorts && version.extracted_cohorts !== version.cohorts.length
          ? ` · ${String(version.extracted_cohorts)} cohorts extracted, merged to ${String(version.cohorts.length)}`
          : ""}
        {shown ? " · shown on the dashboard" : ""}
      </p>
      <ul className="flex flex-col divide-y divide-gray-100 rounded-md border border-gray-200">
        {version.cohorts.map((c) => (
          <li key={c.cohort_id} className="flex items-center gap-3 px-3 py-2">
            <div className="min-w-0 flex-1">
              <p className="line-clamp-2 font-medium text-gray-900" title={c.name}>
                {c.name}
              </p>
              <p className="text-xs text-gray-500">
                {c.key_inclusion.length} key inclusion · {c.key_exclusion.length} key exclusion ·{" "}
                {c.common_count} common
              </p>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                onCriteria(c.cohort_id);
              }}
            >
              Criteria
            </Button>
            <Button
              size="sm"
              variant="outline"
              render={<Link to="/cohorts/$cohortId/patients" params={{ cohortId: c.cohort_id }} />}
            >
              <UsersIcon />
              Find patients
            </Button>
          </li>
        ))}
      </ul>
    </div>
  );
}
