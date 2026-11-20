import { useMutation, useQuery } from "@tanstack/react-query";
import { BotIcon, ListChecksIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { CriteriaList, disagree } from "@/components/criteria-list";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  api,
  type CriteriaItem,
  type CriteriaLLMResult,
  type CriteriaResult,
  type Job,
  runCheck,
  waitForJob,
} from "@/lib/api";
import { llmOptions } from "@/lib/queries";

/**
 * The decision model's per-criterion check of one patient against every line of one cohort: yes means the
 * patient does not clearly fail the line (an inclusion line clearly not met, or an exclusion line clearly met).
 * "Check with LLM" asks the configured LLM endpoint the same about each line (the per-criterion teacher
 * prompt), shown beside the decision model's answers with their disagreements.
 */
export function CriteriaCheck({
  patientId,
  cohortId,
  items,
  autoStart = false,
}: {
  patientId: string;
  cohortId: string;
  items: CriteriaItem[];
  autoStart?: boolean;
}) {
  const [progress, setProgress] = useState("");
  const check = useMutation({
    mutationFn: async () => {
      const { job } = await api.post<{ job: Job }>("/api/criteria", {
        patient_id: patientId,
        cohort_id: cohortId,
      });
      return waitForJob<CriteriaResult>(job.id, (j) => {
        setProgress(j.message);
      });
    },
  });
  const started = useRef(false);
  const { mutate } = check;
  useEffect(() => {
    if (autoStart && !started.current) {
      started.current = true;
      mutate();
    }
  }, [autoStart, mutate]);
  const result = check.data;
  const fails = result?.lines.filter((l) => l.fails) ?? [];
  const llmStatus = useQuery(llmOptions()).data;
  // Disabled until the endpoint's status is known.
  const llmReason = !llmStatus
    ? "Checking the LLM endpoint…"
    : !llmStatus.criteria.available
      ? llmStatus.criteria.reason
      : null;
  const [llmProgress, setLlmProgress] = useState("");
  const llm = useMutation({
    mutationFn: (rerun: boolean) =>
      runCheck<CriteriaLLMResult>(
        "/api/criteria-llm",
        { patient_id: patientId, cohort_id: cohortId, rerun },
        (j) => {
          setLlmProgress(
            `${j.message || j.status}${j.seconds ? ` · ${String(Math.round(j.seconds))} s` : ""}`,
          );
        },
      ),
  });
  const answers = llm.data;
  const llmButton = (
    <Button
      size="sm"
      variant="outline"
      disabled={llm.isPending || llmReason !== null}
      onClick={() => {
        llm.mutate(Boolean(answers));
      }}
    >
      {llm.isPending ? <Spinner /> : <BotIcon />}
      {answers ? "Ask the LLM again" : "Check with LLM"}
    </Button>
  );
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          size="sm"
          variant={result ? "outline" : "default"}
          disabled={check.isPending}
          onClick={() => {
            check.mutate();
          }}
        >
          {check.isPending ? <Spinner /> : <ListChecksIcon />}
          {result ? "Check again" : "Check each criterion"}
        </Button>
        {check.isPending ? (
          <span className="text-sm text-gray-500">{progress || "Checking each criterion…"}</span>
        ) : null}
        {check.isError ? <span className="text-sm text-red-700">{check.error.message}</span> : null}
        {result ? (
          <span className="text-sm text-gray-700">
            {fails.length === 0 ? (
              <>No line is clearly failed ({result.lines.length} checked).</>
            ) : (
              <>
                <strong className="text-red-700">
                  {fails.length} {fails.length === 1 ? "line" : "lines"} clearly failed
                </strong>{" "}
                of {result.lines.length} checked.
              </>
            )}
          </span>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {llmReason === null ? (
          llmButton
        ) : (
          <Tooltip>
            <TooltipTrigger render={<span className="inline-flex" tabIndex={0} />}>
              {llmButton}
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">{llmReason}</TooltipContent>
          </Tooltip>
        )}
        {llm.isPending ? (
          <span className="text-sm text-gray-500">{llmProgress || "Asking the LLM…"}</span>
        ) : null}
        {llm.isError ? <span className="text-sm text-red-700">{llm.error.message}</span> : null}
        {answers ? <LLMSummary answers={answers} decisions={result} /> : null}
        {llmStatus ? (
          <span className="w-full text-xs text-gray-500">{llmStatus.notice}</span>
        ) : null}
      </div>
      <CriteriaList items={items} checks={result?.lines ?? null} llm={answers?.lines ?? null} />
    </div>
  );
}

/** The LLM's per-criterion answers in one line: failures, agreement with the decision model, model and time. */
function LLMSummary({
  answers,
  decisions,
}: {
  answers: CriteriaLLMResult;
  decisions: CriteriaResult | undefined;
}) {
  const byNumber = new Map((decisions?.lines ?? []).map((l) => [l.number, l]));
  const compared = answers.lines.filter((l) => l.fails !== null && byNumber.has(l.number));
  const differ = answers.lines.filter((l) => disagree(byNumber.get(l.number), l));
  const c = answers.counts;
  return (
    <span className="text-sm text-gray-700">
      <strong>LLM:</strong>{" "}
      {c.clearly_fail ? (
        <strong className="text-red-700">
          {c.clearly_fail} {c.clearly_fail === 1 ? "line" : "lines"} clearly failed
        </strong>
      ) : (
        "no line clearly failed"
      )}{" "}
      of {c.lines}
      {c.unanswered ? ` (${String(c.unanswered)} without a valid answer)` : ""}
      {decisions ? (
        <>
          {" · "}agrees with the decision model on {compared.length - differ.length} of{" "}
          {compared.length}
          {differ.length ? (
            <strong className="text-amber-700">
              {" "}
              ({differ.length} {differ.length === 1 ? "disagreement" : "disagreements"})
            </strong>
          ) : null}
        </>
      ) : null}
      <span className="text-xs text-gray-500">
        {" · "}
        {answers.connection.label} · {answers.seconds} s{answers.cached ? " · cached" : ""}
        {c.retries ? ` · ${String(c.retries)} retries` : ""}
      </span>
    </span>
  );
}
