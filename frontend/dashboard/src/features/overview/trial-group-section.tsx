import { Link } from "@tanstack/react-router";
import { cn } from "cn";
import { CircleCheckIcon, FileTextIcon, GlobeIcon, InfoIcon, UsersIcon } from "lucide-react";
import { type ReactNode, useState } from "react";

import { AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  type CohortRow,
  type DrugBrief,
  type OverviewGroup,
  type PendingTrial,
  VERSION_LABELS,
} from "@/lib/api";
import { typeLabels } from "@/lib/format";

import {
  CatalogLink,
  ClassLinks,
  DrugLinks,
  type OpenEvidence,
  PayloadLinks,
} from "./catalog-links";
import { COLUMNS, type ColumnKey, drugsIn, STUDIED_ROLES } from "./search";

interface TrialGroupSectionProps {
  group: OverviewGroup;
  columns: Record<ColumnKey, boolean>;
  labels: Record<string, string>;
  onDetails: (trialId: string) => void;
  onCriteria: (cohortId: string) => void;
  /** Opens a drug's or class's catalog evidence. */
  onEvidence?: OpenEvidence;
}

/** One trial group: its header (as mm-dash's "Group: ... (n trials, m cohorts)") and its cohorts' grid. */
export function TrialGroupSection({
  group,
  columns,
  labels,
  onDetails,
  onCriteria,
  onEvidence,
}: TrialGroupSectionProps) {
  const shown = COLUMNS.filter((c) => columns[c.key]);
  const total = shown.reduce((sum, c) => sum + c.width, 0);
  const trials = new Set(group.rows.map((r) => r.trial.id)).size;
  return (
    <AccordionItem value={group.id} className="border-gray-300 data-open:bg-gray-100">
      <AccordionTrigger className="px-4 py-3 font-normal text-gray-900 hover:bg-gray-50 data-open:hover:bg-gray-200/70">
        <span>
          <strong>Group:</strong> {group.name} ({trials} {trials === 1 ? "trial" : "trials"},{" "}
          {group.rows.length} {group.rows.length === 1 ? "cohort" : "cohorts"})
        </span>
      </AccordionTrigger>
      <AccordionContent className="pb-0">
        {shown.length > 0 ? (
          <Table variant="grid">
            <colgroup>
              {shown.map((c) => (
                <col key={c.key} style={{ width: `${String((c.width / total) * 100)}%` }} />
              ))}
            </colgroup>
            <TableHeader>
              <TableRow>
                {shown.map((c) => (
                  <TableHead key={c.key}>{c.label}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {group.rows.map((row) => (
                <CohortRowView
                  key={row.cohort_id}
                  row={row}
                  columns={columns}
                  labels={labels}
                  onDetails={onDetails}
                  onCriteria={onCriteria}
                  onEvidence={onEvidence}
                />
              ))}
            </TableBody>
          </Table>
        ) : (
          <p className="px-4 py-3 text-sm text-muted-foreground">No columns selected.</p>
        )}
      </AccordionContent>
    </AccordionItem>
  );
}

function CohortRowView({
  row,
  columns,
  labels,
  onDetails,
  onCriteria,
  onEvidence,
}: {
  row: CohortRow;
  columns: Record<ColumnKey, boolean>;
  labels: Record<string, string>;
  onDetails: (trialId: string) => void;
  onCriteria: (cohortId: string) => void;
  onEvidence: OpenEvidence | undefined;
}) {
  return (
    <TableRow
      className={row.trial.is_open ? "bg-white" : "bg-gray-100 opacity-60"}
      data-cohort-id={row.cohort_id}
    >
      {columns.irbPi ? <IrbPiCell row={row} onDetails={onDetails} /> : null}
      {columns.cohort ? <CohortCell row={row} labels={labels} /> : null}
      {columns.eligibility ? <EligibilityCell row={row} onCriteria={onCriteria} /> : null}
      {columns.drugMechanism ? <DrugMechanismCell row={row} onEvidence={onEvidence} /> : null}
      {columns.contacts ? <ContactsCell row={row} /> : null}
      {columns.patients ? <PatientsCell row={row} /> : null}
    </TableRow>
  );
}

function Cell({ children }: { children: ReactNode }) {
  return (
    <TableCell className="h-px">
      <div className="flex h-full flex-col">{children}</div>
    </TableCell>
  );
}

export function SourceIcon({ version }: { version: CohortRow["version"] }) {
  const Icon = version === "protocol" ? FileTextIcon : GlobeIcon;
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span
            className={cn(
              "inline-flex",
              version === "protocol" ? "text-green-600" : "text-brand-light-blue",
            )}
            aria-label={`Cohorts from the ${VERSION_LABELS[version]} version`}
          />
        }
      >
        <Icon className="size-3 shrink-0" />
      </TooltipTrigger>
      <TooltipContent>
        {version === "protocol"
          ? "Cohorts extracted from the trial's protocol"
          : "Cohorts extracted from the ClinicalTrials.gov listing (protocol not processed yet)"}
      </TooltipContent>
    </Tooltip>
  );
}

function IrbPiCell({ row, onDetails }: { row: CohortRow; onDetails: (trialId: string) => void }) {
  const t = row.trial;
  return (
    <Cell>
      <div className="flex items-center gap-1 font-bold text-gray-900">
        <SourceIcon version={row.version} />
        {t.protocol_no || t.nct_id}
      </div>
      {t.site_pi ? <div className="font-semibold">{t.site_pi}</div> : null}
      <div>{t.short_title}</div>
      <div className="text-xs text-gray-500">{[t.phase, t.nct_id].filter(Boolean).join(" · ")}</div>
      <div className="mt-2 text-center print:hidden">
        <button
          type="button"
          className="link text-xs"
          onClick={() => {
            onDetails(t.id);
          }}
        >
          Trial details
        </button>
      </div>
    </Cell>
  );
}

function CohortCell({ row, labels }: { row: CohortRow; labels: Record<string, string> }) {
  const types = typeLabels(row.cancer_types, labels);
  return (
    <Cell>
      <div className="line-clamp-4 font-bold text-gray-900" title={row.name}>
        {row.name}
      </div>
      {types.length ? (
        <div className="mt-1">
          <strong>Cancer:</strong> {types.join(", ")}
        </div>
      ) : null}
      {row.merged_from > 1 ? (
        <div className="text-xs text-gray-500">
          Combines {row.merged_from} cohorts with the same criteria
        </div>
      ) : null}
      {row.group_reason ? (
        <div className="mt-auto flex justify-end pt-2 print:hidden">
          <Tooltip>
            <TooltipTrigger
              render={
                <span
                  className="inline-flex text-gray-300 hover:text-gray-500"
                  aria-label="Why this group"
                />
              }
            >
              <InfoIcon className="size-3.5" />
            </TooltipTrigger>
            <TooltipContent className="max-w-sm">
              In this group by its {row.group_reason}
            </TooltipContent>
          </Tooltip>
        </div>
      ) : null}
    </Cell>
  );
}

const SHOWN_LINES = 4;

function CriteriaLines({ lines, label }: { lines: string[]; label: string }) {
  const [all, setAll] = useState(false);
  if (lines.length === 0) return null;
  const shown = all ? lines : lines.slice(0, SHOWN_LINES);
  return (
    <div className="flex flex-col gap-0.5">
      <strong className="text-gray-700">{label}</strong>
      <ul className="flex list-none flex-col gap-0.5">
        {shown.map((line, i) => (
          <li key={i} className={cn("whitespace-pre-wrap", !all && "line-clamp-3")}>
            {line}
          </li>
        ))}
      </ul>
      {lines.length > SHOWN_LINES || (!all && lines.some((l) => l.length > 240)) ? (
        <button
          type="button"
          className="link self-start text-xs print:hidden"
          onClick={() => {
            setAll((v) => !v);
          }}
        >
          {all
            ? "Show less"
            : lines.length > SHOWN_LINES
              ? `+${String(lines.length - SHOWN_LINES)} more`
              : "Show all"}
        </button>
      ) : null}
    </div>
  );
}

function EligibilityCell({
  row,
  onCriteria,
}: {
  row: CohortRow;
  onCriteria: (cohortId: string) => void;
}) {
  return (
    <Cell>
      <div className="flex flex-col gap-2">
        <CriteriaLines lines={row.key_inclusion} label="Inclusion" />
        <CriteriaLines lines={row.key_exclusion} label="Exclusion" />
      </div>
      <div className="mt-auto flex items-center gap-3 pt-2 text-xs print:hidden">
        <button
          type="button"
          className="link"
          onClick={() => {
            onCriteria(row.cohort_id);
          }}
        >
          All {row.line_count} criteria
        </button>
        {row.common_count ? (
          <span className="text-gray-400">{row.common_count} common criteria not shown</span>
        ) : null}
      </div>
    </Cell>
  );
}

/** "Label: value" (mm-dash's labeled lines); nothing when there is no value. */
function Labeled({ label, children }: { label: string; children: ReactNode }) {
  if (children === null || children === undefined || children === "") return null;
  return (
    <div>
      <strong>{label}:</strong> {children}
    </div>
  );
}

const SHOWN_DRUGS = 3;

/** Other drugs of the trial in one muted line: "vs ..." (control), "with ..." (background). */
function OtherDrugs({
  prefix,
  drugs,
  onEvidence,
}: {
  prefix: string;
  drugs: DrugBrief[];
  onEvidence: OpenEvidence | undefined;
}) {
  if (drugs.length === 0) return null;
  return (
    <div className="line-clamp-2 text-xs text-gray-500" title={drugs.map((d) => d.name).join(", ")}>
      {prefix} <DrugLinks drugs={drugs} onOpen={onEvidence} />
    </div>
  );
}

/**
 * mm-dash's Drug / Mechanism cell, from the trial's drugs in the GoodOption catalog: each investigational (or
 * role-uncertain) drug's name, its most specific mechanism classes and an ADC's payload, then its comparators
 * and background therapy. The catalog lists a trial's drugs, not a cohort's, so every cohort of a trial shows them.
 * Each drug and class name opens its catalog evidence.
 */
function DrugMechanismCell({
  row,
  onEvidence,
}: {
  row: CohortRow;
  onEvidence: OpenEvidence | undefined;
}) {
  const [all, setAll] = useState(false);
  const studied = drugsIn(row, STUDIED_ROLES);
  const control = drugsIn(row, ["control"]);
  const background = drugsIn(row, ["background"]);
  if (studied.length + control.length + background.length === 0) {
    return (
      <Cell>
        <span className="text-gray-400">—</span>
      </Cell>
    );
  }
  const shown = all ? studied : studied.slice(0, SHOWN_DRUGS);
  return (
    <Cell>
      <div className="flex flex-col gap-1.5">
        {shown.map((d) => (
          <div key={d.drug_id}>
            <Labeled label="Name">
              <CatalogLink kind="drug" id={d.drug_id} onOpen={onEvidence} variant="link">
                {d.name}
              </CatalogLink>
              {d.role === "uncertain" ? (
                <span className="text-xs text-gray-500"> (role uncertain)</span>
              ) : null}
            </Labeled>
            <Labeled label="Type">
              {d.mechanism_types.length ? (
                <ClassLinks names={d.mechanism_types} ids={d.class_ids} onOpen={onEvidence} />
              ) : null}
            </Labeled>
            <Labeled label="Payload">
              {d.payload ? <PayloadLinks drug={d} onOpen={onEvidence} /> : null}
            </Labeled>
          </div>
        ))}
      </div>
      {studied.length > SHOWN_DRUGS ? (
        <button
          type="button"
          className="link self-start text-xs print:hidden"
          onClick={() => {
            setAll((v) => !v);
          }}
        >
          {all ? "Show less" : `+${String(studied.length - SHOWN_DRUGS)} more`}
        </button>
      ) : null}
      <div className={cn("flex flex-col gap-0.5", studied.length > 0 && "mt-1.5")}>
        <OtherDrugs prefix="vs" drugs={control} onEvidence={onEvidence} />
        <OtherDrugs prefix="with" drugs={background} onEvidence={onEvidence} />
      </div>
      <div className="mt-auto flex justify-end pt-2 print:hidden">
        <Tooltip>
          <TooltipTrigger
            render={
              <span
                className="inline-flex text-gray-300 hover:text-gray-500"
                aria-label="Where the drugs come from"
              />
            }
          >
            <InfoIcon className="size-3.5" />
          </TooltipTrigger>
          <TooltipContent className="max-w-sm">
            The trial's drugs, from the GoodOption catalog (LLM-assisted, NCIt classes); verify
            against the protocol. The catalog lists a trial's drugs, not a cohort's.
          </TooltipContent>
        </Tooltip>
      </div>
    </Cell>
  );
}

function ContactsCell({ row }: { row: CohortRow }) {
  const c = row.trial.contact;
  return (
    <Cell>
      {c.name ? <div className="font-semibold text-gray-900">{c.name}</div> : null}
      {c.role ? <div className="text-xs">{c.role}</div> : null}
      {c.email ? (
        <a className="link text-xs break-all" href={`mailto:${c.email}`}>
          {c.email}
        </a>
      ) : null}
      {row.trial.overall_pi && row.trial.overall_pi !== row.trial.site_pi ? (
        <div className="mt-2 text-xs">
          <strong>Overall PI:</strong> {row.trial.overall_pi}
        </div>
      ) : null}
    </Cell>
  );
}

function PatientsCell({ row }: { row: CohortRow }) {
  return (
    <Cell>
      <Button
        size="sm"
        variant="outline"
        className="print:hidden"
        render={<Link to="/cohorts/$cohortId/patients" params={{ cohortId: row.cohort_id }} />}
      >
        <UsersIcon />
        Find patients
      </Button>
    </Cell>
  );
}

const STATE_TEXT: Record<string, string> = {
  not_treatment: "Not a cancer treatment trial (cohort screen)",
  not_run: "Not extracted: no protocol or listing",
  pending: "Extraction in progress",
  extracting: "Extraction in progress",
  not_run_protocol: "",
};

/** Trials whose cohorts aren't extracted (yet), after the groups. */
export function PendingSection({
  trials,
  onDetails,
}: {
  trials: PendingTrial[];
  onDetails: (trialId: string) => void;
}) {
  return (
    <AccordionItem value="pending" className="border-gray-300 data-open:bg-gray-100">
      <AccordionTrigger className="px-4 py-3 font-normal text-gray-600 hover:bg-gray-50">
        <span>
          <strong>Trials without extracted cohorts</strong> ({trials.length})
        </span>
      </AccordionTrigger>
      <AccordionContent className="pb-0">
        <Table variant="grid">
          <TableHeader>
            <TableRow>
              <TableHead className="w-1/4">IRB# / PI</TableHead>
              <TableHead>Title</TableHead>
              <TableHead className="w-1/3">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {trials.map((t) => (
              <TableRow key={t.id} className="bg-white">
                <TableCell>
                  <div className="font-bold text-gray-900">{t.protocol_no || t.nct_id}</div>
                  {t.site_pi ? <div className="font-semibold">{t.site_pi}</div> : null}
                  <button
                    type="button"
                    className="link text-xs"
                    onClick={() => {
                      onDetails(t.id);
                    }}
                  >
                    Trial details
                  </button>
                </TableCell>
                <TableCell>
                  {t.short_title}
                  <div className="text-xs text-gray-500">
                    {[t.phase, t.nct_id].filter(Boolean).join(" · ")}
                  </div>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-1">
                    {t.state === "not_treatment" ? (
                      <CircleCheckIcon className="size-3 text-gray-400" />
                    ) : null}
                    {STATE_TEXT[t.state] ?? t.state}
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </AccordionContent>
    </AccordionItem>
  );
}
