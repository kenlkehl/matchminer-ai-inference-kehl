import { cn } from "cn";
import { CircleCheckIcon, CircleHelpIcon, CircleXIcon } from "lucide-react";

import type { CriteriaItem, CriterionLine, LLMCriterionLine } from "@/lib/api";
import { percent } from "@/lib/format";

/** Whether the decision model and the LLM both answered a line, differently. */
export function disagree(check: CriterionLine | undefined, llm: LLMCriterionLine | undefined) {
  return Boolean(check && llm && llm.fails !== null && check.fails !== llm.fails);
}

/**
 * A cohort's criteria lines under Inclusion and Exclusion, common ("boilerplate") lines marked, and,
 * when a per-criterion check ran, each line's answer: a clear failure, or no clear failure. With
 * `llm`, the LLM's answer and reason follow the decision model's, and a line they answer differently
 * is highlighted.
 */
export function CriteriaList({
  items,
  checks,
  llm,
  showCommon = true,
}: {
  items: CriteriaItem[];
  checks?: CriterionLine[] | null;
  llm?: LLMCriterionLine[] | null;
  showCommon?: boolean;
}) {
  const byNumber = new Map((checks ?? []).map((c) => [c.number, c]));
  const llmByNumber = new Map((llm ?? []).map((c) => [c.number, c]));
  const sections = (["inclusion", "exclusion"] as const).map((kind) => ({
    kind,
    items: items.filter(
      (i) =>
        (i.criterion_type ??
          (i.heading.toLowerCase().startsWith("excl") ? "exclusion" : "inclusion")) === kind,
    ),
  }));
  return (
    <div className="flex flex-col gap-4">
      {sections.map((section) =>
        section.items.length ? (
          <section key={section.kind}>
            <h4 className="mb-1 text-sm font-bold capitalize">{section.kind} criteria</h4>
            <ol className="flex flex-col divide-y divide-gray-100">
              {section.items
                .filter((i) => showCommon || !i.boilerplate)
                .map((item) => {
                  const check = byNumber.get(item.number);
                  const answer = llmByNumber.get(item.number);
                  const differ = disagree(check, answer);
                  return (
                    <li
                      key={item.number}
                      className={cn(
                        "flex items-start gap-2 py-1.5 text-sm",
                        differ && "-mx-2 rounded bg-amber-50 px-2 ring-1 ring-amber-200",
                      )}
                    >
                      {checks ? <CheckMark check={check} /> : null}
                      {llm ? <LLMMark line={answer} /> : null}
                      <div className="flex flex-1 flex-col gap-0.5">
                        <span
                          className={cn(
                            "whitespace-pre-wrap",
                            item.boilerplate ? "text-gray-500" : "text-gray-800",
                            (check?.fails ?? answer?.fails) && "font-medium text-red-800",
                          )}
                        >
                          {item.text}
                        </span>
                        {answer ? <LLMReason line={answer} differ={differ} /> : null}
                      </div>
                      {item.boilerplate ? (
                        <span className="shrink-0 rounded bg-gray-100 px-1.5 py-0.5 text-[11px] text-gray-500">
                          Common
                        </span>
                      ) : null}
                    </li>
                  );
                })}
            </ol>
          </section>
        ) : null,
      )}
    </div>
  );
}

function LLMMark({ line }: { line: LLMCriterionLine | undefined }) {
  if (!line) return <span className="mt-0.5 size-4 shrink-0" />;
  const Icon = line.fails === null ? CircleHelpIcon : line.fails ? CircleXIcon : CircleCheckIcon;
  const tone =
    line.fails === null ? "text-gray-400" : line.fails ? "text-red-600" : "text-green-600";
  return (
    <span
      title={
        line.fails === null
          ? `LLM: no valid answer (${line.parse_status})`
          : `LLM: ${line.fails ? "clearly fails" : "no clear failure"} (${line.status ?? ""})`
      }
      className={cn("mt-0.5 shrink-0 rounded-full ring-1 ring-offset-1", tone, "ring-current/30")}
    >
      <Icon className="size-4" />
    </span>
  );
}

function LLMReason({ line, differ }: { line: LLMCriterionLine; differ: boolean }) {
  return (
    <div className="text-xs text-gray-600">
      <span className="font-medium">
        LLM{line.status ? ` (${line.status.replace(/_/g, " ")})` : ""}:
      </span>{" "}
      {line.answer === null ? `no valid answer: ${line.parse_status}` : line.reason}
      {differ ? (
        <span className="ml-1 rounded bg-amber-200 px-1 py-0.5 text-[11px] font-medium text-amber-900">
          differs from the decision model
        </span>
      ) : null}
      {line.reasoning ? (
        <details className="mt-0.5">
          <summary className="cursor-pointer text-gray-400 hover:text-gray-600">Reasoning</summary>
          <p className="mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap text-gray-500">
            {line.reasoning}
          </p>
        </details>
      ) : null}
    </div>
  );
}

function CheckMark({ check }: { check: CriterionLine | undefined }) {
  if (!check) return <span className="mt-0.5 size-4 shrink-0" />;
  return check.fails ? (
    <span
      title={`Clearly fails (P(no clear failure) ${percent(check.p)})`}
      className="mt-0.5 shrink-0 text-red-600"
    >
      <CircleXIcon className="size-4" />
    </span>
  ) : (
    <span
      title={`No clear failure (${percent(check.p)})`}
      className="mt-0.5 shrink-0 text-green-600"
    >
      <CircleCheckIcon className="size-4" />
    </span>
  );
}
