import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { CircleCheckIcon, CircleXIcon, RotateCwIcon, ScaleIcon } from "lucide-react";
import { type ReactNode, useSyncExternalStore } from "react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
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
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { type GoodOptionResult, type LLMStatus, runCheck, type TrialSummary } from "@/lib/api";
import { percent, when } from "@/lib/format";
import { llmOptions } from "@/lib/queries";

import { CatalogLink, ClassLinks, type OpenEvidence } from "../overview/catalog-links";
import { STUDIED_ROLES } from "../overview/search";

// ---------------------------------------------------------------- the runs (kept while the tab is open)

interface Run {
  status: "running" | "done" | "error";
  message: string;
  result?: GoodOptionResult;
  error?: string;
}

const runs = new Map<string, Run>();
const listeners = new Set<() => void>();
const keyOf = (patientId: string, trialId: string) => `${patientId}|${trialId}`;

function setRun(key: string, run: Run) {
  runs.set(key, run);
  for (const listener of listeners) listener();
}

export function useGoodOptionRun(patientId: string, trialId: string): Run | undefined {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => runs.get(keyOf(patientId, trialId)),
  );
}

/** Run the GoodOption check of a patient and a trial (its cached answer when the inputs are unchanged). */
export function startGoodOption(patientId: string, trialId: string, rerun = false) {
  const key = keyOf(patientId, trialId);
  if (runs.get(key)?.status === "running") return;
  setRun(key, { status: "running", message: "starting" });
  runCheck<GoodOptionResult>(
    "/api/good-option",
    { patient_id: patientId, trial_id: trialId, rerun },
    (job) => {
      setRun(key, {
        status: "running",
        message: `${job.message || job.status}${job.seconds ? ` · ${String(Math.round(job.seconds))} s` : ""}`,
      });
    },
  ).then(
    (result) => {
      setRun(key, { status: "done", message: "", result });
    },
    (error: unknown) => {
      setRun(key, {
        status: "error",
        message: "",
        error: error instanceof Error ? error.message : String(error),
      });
    },
  );
}

/** Why the trial can't be checked (the button is then disabled), or null. */
export function goodOptionUnavailable(
  trial: TrialSummary,
  status: LLMStatus | undefined,
): string | null {
  if (!status) return "Checking the LLM endpoint…";
  if (!status.good_option.available) return status.good_option.reason;
  if (trial.drugs === undefined) return "This server predates the trials' drug information.";
  const studied: readonly string[] = STUDIED_ROLES;
  if (!trial.drugs.some((d) => studied.includes(d.role)))
    return "The GoodOption catalog lists no investigational drug to score for this trial.";
  return null;
}

/** The trial card's Good option check: its state (running, the score, an error) and the dialog. */
export function GoodOptionButton({
  patientId,
  trial,
  onOpen,
}: {
  patientId: string;
  trial: TrialSummary;
  onOpen: () => void;
}) {
  const status = useQuery(llmOptions()).data;
  const run = useGoodOptionRun(patientId, trial.id);
  const reason = goodOptionUnavailable(trial, status);
  const result = run?.result;
  const button = (
    <Button
      size="sm"
      variant="outline"
      disabled={reason !== null}
      onClick={(event) => {
        event.stopPropagation();
        if (!run || run.status === "error") startGoodOption(patientId, trial.id);
        onOpen();
      }}
    >
      {run?.status === "running" ? <Spinner /> : <ScaleIcon />}
      {result?.status === "ok" && result.points !== null && result.max_points !== null
        ? `Good option: ${String(result.points)}/${String(result.max_points)}`
        : "Good option check"}
    </Button>
  );
  return (
    <div className="flex flex-col items-stretch gap-1">
      {reason === null ? (
        button
      ) : (
        <Tooltip>
          <TooltipTrigger render={<span className="inline-flex flex-col" tabIndex={0} />}>
            {button}
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">{reason}</TooltipContent>
        </Tooltip>
      )}
      {run?.status === "running" ? (
        <span className="max-w-40 text-xs text-gray-500">{run.message}</span>
      ) : run?.status === "error" ? (
        <span className="max-w-40 text-xs text-red-700" title={run.error}>
          Check failed; click to try again
        </span>
      ) : result && result.status !== "ok" ? (
        <span className="max-w-40 text-xs text-gray-500">{result.status_text}</span>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------- the dialog

const CAVEAT =
  "GoodOption catalog (LLM-assisted synthesis of the cited sources; verify before use). A ranking signal for review, not an eligibility decision or treatment recommendation.";

/**
 * The GoodOption check of one patient and one trial (`?good=<trialId>` on the trials-for-a-patient
 * view): the trial's score, each scored drug's four rubric answers with their rationales, the
 * uncertainties, the rubric in the package's words, the reasoning, and the model and timing.
 * `children` (the catalog evidence dialog) render inside it, so they stack over it.
 */
export function GoodOptionDialog({
  patientId,
  trialId,
  trial,
  onClose,
  onEvidence,
  children,
}: {
  patientId: string;
  trialId: string;
  trial: TrialSummary | undefined;
  onClose: () => void;
  onEvidence: OpenEvidence;
  children?: ReactNode;
}) {
  const status = useQuery(llmOptions()).data;
  const run = useGoodOptionRun(patientId, trialId);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-3xl" bodyClassName="gap-4">
        <DialogHeader className="gap-2">
          <DialogTitle>
            Good option check: {trial ? trial.protocol_no || trial.nct_id : trialId}
          </DialogTitle>
          <DialogDescription className="text-gray-600">
            {[trial?.short_title, `patient ${patientId}`].filter(Boolean).join(" · ")}
          </DialogDescription>
          <p className="rounded-md bg-amber-50 px-3 py-1.5 text-xs text-amber-900">
            {CAVEAT} {status?.notice}
          </p>
        </DialogHeader>
        {!run ? (
          <div>
            <Button
              onClick={() => {
                startGoodOption(patientId, trialId);
              }}
            >
              <ScaleIcon />
              Run the Good option check
            </Button>
          </div>
        ) : run.status === "running" ? (
          <p className="flex items-center gap-2 text-sm text-gray-600">
            <Spinner /> {run.message}
          </p>
        ) : run.status === "error" ? (
          <p className="text-sm text-red-700">The check failed: {run.error}</p>
        ) : run.result ? (
          <ResultView result={run.result} status={status} onEvidence={onEvidence} />
        ) : null}
        <DialogFooter className="pt-1">
          {run && run.status !== "running" ? (
            <Button
              variant="outline"
              onClick={() => {
                startGoodOption(patientId, trialId, true);
              }}
            >
              <RotateCwIcon />
              Run again
            </Button>
          ) : null}
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
        {children}
      </DialogContent>
    </Dialog>
  );
}

function ResultView({
  result: r,
  status,
  onEvidence,
}: {
  result: GoodOptionResult;
  status: LLMStatus | undefined;
  onEvidence: OpenEvidence;
}) {
  const questions = new Map((status?.rubric.criteria ?? []).map((c) => [c.criterion, c.question]));
  return (
    <>
      <section className="flex flex-col gap-1">
        {r.status === "ok" ? (
          <p className="text-base">
            <strong>
              {r.points}/{r.max_points} points
            </strong>{" "}
            ({r.score !== null ? percent(r.score) : "?"}) over {r.drug_count}{" "}
            {r.drug_count === 1 ? "scored drug" : "scored drugs"}
            {r.patient_disease_type ? (
              <span className="text-sm text-gray-600">
                {" "}
                · disease as judged: {r.patient_disease_type}
              </span>
            ) : null}
          </p>
        ) : (
          <p className="font-medium text-red-800">{r.status_text}</p>
        )}
        <p className="text-xs text-gray-500">{status?.rubric.score_meaning}</p>
        {r.parse_error ? (
          <p className="text-xs text-red-700">Last parse error: {r.parse_error}</p>
        ) : null}
      </section>
      {r.assessments.map((a) => (
        <section key={a.drug_name} className="rounded-md border border-gray-200">
          <div className="flex flex-wrap items-baseline gap-x-2 border-b border-gray-100 px-3 py-2">
            <CatalogLink
              kind="drug"
              id={a.drug_id ?? undefined}
              onOpen={onEvidence}
              variant="link"
              className="font-semibold"
            >
              {a.drug_name}
            </CatalogLink>
            {a.role ? <span className="text-xs text-gray-500">{a.role}</span> : null}
            <Badge variant="secondary" className="ml-auto self-center">
              {a.points}/4
            </Badge>
            {a.mechanism_types.length ? (
              <p className="w-full text-sm text-gray-700">
                <strong>Type:</strong>{" "}
                <ClassLinks names={a.mechanism_types} ids={a.class_ids} onOpen={onEvidence} />
              </p>
            ) : null}
            {a.targeted_biomarkers.length ? (
              <p className="w-full text-xs text-gray-600">
                Targets as judged: {a.targeted_biomarkers.join(", ")}
              </p>
            ) : null}
          </div>
          <ol className="flex flex-col divide-y divide-gray-100">
            {a.criteria.map((c, i) => (
              <li key={c.criterion} className="flex gap-2 px-3 py-2 text-sm">
                {c.point ? (
                  <CircleCheckIcon className="mt-0.5 size-4 shrink-0 text-green-600" />
                ) : (
                  <CircleXIcon className="mt-0.5 size-4 shrink-0 text-gray-400" />
                )}
                <div className="flex flex-col gap-0.5">
                  <span className="font-medium text-gray-900">
                    {i + 1}. {questions.get(c.criterion) ?? c.criterion}
                  </span>
                  <span className="text-gray-700">{c.rationale}</span>
                </div>
              </li>
            ))}
          </ol>
        </section>
      ))}
      {r.uncertainties.length ? (
        <section>
          <h3 className="mb-1 text-sm font-bold">Key uncertainties</h3>
          <ul className="list-disc pl-5 text-sm text-gray-700">
            {r.uncertainties.map((u, i) => (
              <li key={i}>{u}</li>
            ))}
          </ul>
        </section>
      ) : null}
      <Accordion multiple className="overflow-hidden rounded-md border border-gray-200">
        {status ? (
          <AccordionItem value="rubric">
            <AccordionTrigger className="px-3 py-2 text-sm font-semibold">
              The rubric, in the inference package's words
            </AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-2 px-3 pb-3 text-xs whitespace-pre-wrap text-gray-700">
                <p>{status.rubric.intro}</p>
                {status.rubric.criteria.map((c, i) => (
                  <p key={c.criterion}>
                    <strong>
                      {i + 1}. {c.criterion}: {c.question}
                    </strong>
                    {"\n"}
                    {c.text}
                  </p>
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        ) : null}
        {r.reasoning ? (
          <AccordionItem value="reasoning">
            <AccordionTrigger className="px-3 py-2 text-sm font-semibold">
              The model's reasoning
            </AccordionTrigger>
            <AccordionContent>
              <p className="max-h-96 overflow-y-auto px-3 pb-3 text-xs whitespace-pre-wrap text-gray-700">
                {r.reasoning}
              </p>
            </AccordionContent>
          </AccordionItem>
        ) : null}
      </Accordion>
      <p className={cn("text-xs text-gray-500")}>
        {r.connection.label} · {r.seconds} s{r.cached ? " · cached answer" : ""} · checked{" "}
        {when(r.checked_at)} · parse attempts {r.parse_attempts ?? "not reported"}
        {r.max_parse_attempts ? ` of ${String(r.max_parse_attempts)}` : ""}
        {r.reasoning_off_fallback ? " (+1 without thinking)" : ""}
        {r.reasoning_off_fallback_used ? ", the last without thinking" : ""}
        {r.finish_reason ? ` · finish ${r.finish_reason}` : ""}
        {r.context.context_tokens
          ? ` · evidence packed for ${r.context.context_tokens.toLocaleString()} tokens (${r.context.context_source ?? ""})`
          : ""}
        {r.context.context_warning ? ` · ${r.context.context_warning}` : ""}
      </p>
    </>
  );
}
