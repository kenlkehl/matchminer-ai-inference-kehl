import { useQuery } from "@tanstack/react-query";
import { getRouteApi, Link, useRouter } from "@tanstack/react-router";
import { ArrowLeftIcon, PanelRightCloseIcon, PanelRightOpenIcon } from "lucide-react";
import { Fragment, useMemo, useRef, useState } from "react";
import type { PanelImperativeHandle } from "react-resizable-panels";

import { CriteriaList } from "@/components/criteria-list";
import { PageHeader } from "@/components/page-header";
import { SidebarSlot } from "@/components/shell/sidebar-slot";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Spinner } from "@/components/ui/spinner";
import {
  api,
  type CohortDetail,
  type PatientHit,
  type PatientsForCohortResult,
  VERSION_LABELS,
} from "@/lib/api";
import { patientLine } from "@/lib/format";
import { metaOptions } from "@/lib/queries";

import { CriteriaCheck } from "./criteria-check";
import { PatientDialog } from "./patient-dialog";
import { byScore, ExclusionLine, likelyExcluded, MatchBadge, Requirements } from "./score-view";
import { useScoring } from "./use-scoring";

const route = getRouteApi("/_app/cohorts/$cohortId/patients");
const SELECT = "h-8 w-full rounded-md border border-white/20 bg-white/10 px-2 text-xs text-white";

/**
 * "Patients for this cohort": the synthetic patients nearest the cohort's key criteria in TrialSpace, the
 * first scored by the decision model as the page opens, with the cohort's criteria in a resizable panel.
 */
export function PatientsForCohortPage() {
  const { cohortId } = route.useParams();
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const meta = useQuery(metaOptions()).data;
  const k = search.k ?? meta?.default_k ?? 50;
  const result = useQuery({
    queryKey: ["patients-for-cohort", cohortId, search.cancer_type ?? "", search.sex ?? "", k],
    queryFn: () =>
      api.post<PatientsForCohortResult>("/api/match/patients-for-cohort", {
        cohort_id: cohortId,
        k,
        cancer_type: search.cancer_type ?? "",
        sex: search.sex ?? "",
      }),
    staleTime: Infinity,
    retry: (count, error) => count < 30 && (error as { status?: number }).status === 409,
    retryDelay: 5_000,
  });
  const data = result.data;
  const rows = useMemo(
    () => (data ? data.results.map((r) => ({ ...r, id: r.patient.id })) : null),
    [data],
  );
  const scoring = useScoring(
    "patients_for_cohort",
    data?.query.cohort_id ?? null,
    rows,
    meta?.auto_score_n ?? 10,
  );
  const [hideExcluded, setHideExcluded] = useState(false);
  const [open, setOpen] = useState<{ id: string; tab: "pair" | "summary" } | null>(null);
  const router = useRouter();

  const shown = rows
    ? byScore(rows, scoring.scores).filter(
        (r) => !(hideExcluded && likelyExcluded(scoring.scores[r.id])),
      )
    : [];
  const unscored = rows?.filter((r) => !scoring.scores[r.id]?.match).map((r) => r.id) ?? [];
  const cohort = data?.query;

  return (
    <div className="flex flex-col gap-4">
      <SidebarSlot>
        <div className="mt-2 flex flex-col gap-4 border-t border-white/20 pt-4 pb-4">
          <h2 className="text-sm font-semibold">Patients</h2>
          <label className="flex flex-col gap-1 text-xs text-gray-200">
            Cancer type
            <select
              className={SELECT}
              value={search.cancer_type ?? ""}
              onChange={(e) => {
                void navigate({
                  search: (prev) => ({ ...prev, cancer_type: e.target.value || undefined }),
                });
              }}
            >
              <option value="">Any</option>
              {meta?.cancer_types
                .filter((t) => t.group !== "Basket")
                .map((t) => (
                  <option key={t.key} value={t.key}>
                    {t.label}
                  </option>
                ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-gray-200">
            Sex
            <select
              className={SELECT}
              value={search.sex ?? ""}
              onChange={(e) => {
                void navigate({
                  search: (prev) => ({ ...prev, sex: e.target.value || undefined }),
                });
              }}
            >
              <option value="">Any</option>
              <option value="female">Female</option>
              <option value="male">Male</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-gray-200">
            Patients retrieved
            <select
              className={SELECT}
              value={k}
              onChange={(e) => {
                void navigate({ search: (prev) => ({ ...prev, k: Number(e.target.value) }) });
              }}
            >
              {[25, 50, 100, 200].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <Label className="gap-1.5 text-xs font-normal text-gray-200">
            <Checkbox
              className="size-3.5"
              checked={hideExcluded}
              onCheckedChange={setHideExcluded}
            />
            Hide patients likely excluded
          </Label>
        </div>
      </SidebarSlot>
      <PageHeader title="Find Patients for a Cohort">
        <p>
          The synthetic patients whose summaries are nearest this cohort's key criteria
          (TrialSpace), ranked by the MatchMiner-AI decision model as they are scored. Matches are
          screening signals for review, not eligibility determinations.
        </p>
      </PageHeader>
      <ResizablePanelGroup
        orientation="horizontal"
        className="-mr-6"
        style={{ width: "calc(100% + 1.5rem)", height: "auto", overflow: "visible" }}
      >
        <ResizablePanel id="results" minSize="40%" style={{ overflow: "visible" }}>
          <div className="flex flex-col gap-4 pr-6">
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  router.history.back();
                }}
              >
                <ArrowLeftIcon />
                Back
              </Button>
              {cohort ? (
                <Button
                  variant="outline"
                  size="sm"
                  render={
                    <Link
                      to="/"
                      search={{ center: cohort.trial.center_id, cohort: cohort.cohort_id }}
                    />
                  }
                >
                  {cohort.trial.center_short_name || cohort.trial.center} dashboard
                </Button>
              ) : null}
            </div>
            {result.isPending ||
            (result.isError && (result.error as { status?: number }).status === 409) ? (
              <p className="flex items-center gap-2 text-sm text-gray-600">
                <Spinner />
                {result.isError ? result.error.message : "Finding patients…"}
              </p>
            ) : result.isError ? (
              <p className="text-sm text-red-700">{result.error.message}</p>
            ) : data ? (
              <section className="flex flex-col gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-xl font-bold">Matched Patients</h2>
                  <Badge variant="secondary">
                    {shown.length} of {data.searched.toLocaleString()} synthetic patients
                  </Badge>
                  <ScoringStatus scoring={scoring} />
                </div>
                <ol className="flex flex-col gap-3">
                  {shown.map((row) => (
                    <PatientCard
                      key={row.id}
                      row={row}
                      scores={scoring.scores[row.id]}
                      scoring={scoring.running.includes(row.id)}
                      onOpen={(tab) => {
                        setOpen({ id: row.id, tab });
                      }}
                    />
                  ))}
                </ol>
                {unscored.length ? (
                  <Button
                    variant="outline"
                    className="self-start"
                    disabled={scoring.running.length > 0}
                    onClick={() => {
                      scoring.score(unscored.slice(0, 10));
                    }}
                  >
                    Score the next {Math.min(10, unscored.length)} patients
                  </Button>
                ) : null}
              </section>
            ) : null}
          </div>
        </ResizablePanel>
        {cohort ? <CohortPanel cohort={cohort} /> : null}
      </ResizablePanelGroup>
      {open && cohort ? (
        <PatientDialog
          patientId={open.id}
          defaultTab={open.tab}
          onClose={() => {
            setOpen(null);
          }}
          pairTab={{
            label: "Criteria check",
            content: (
              <CriteriaCheck
                patientId={open.id}
                cohortId={cohort.cohort_id}
                items={cohort.items}
                autoStart
              />
            ),
          }}
        />
      ) : null}
    </div>
  );
}

export function ScoringStatus({ scoring }: { scoring: ReturnType<typeof useScoring> }) {
  if (scoring.error)
    return <span className="text-sm text-red-700">Scoring failed: {scoring.error}</span>;
  if (!scoring.progress) return null;
  return (
    <span className="flex items-center gap-1.5 text-sm text-gray-500">
      <Spinner className="size-3.5" />
      Scoring the top {scoring.running.length} with the decision model
      {scoring.progress.total
        ? ` (${String(Math.round((100 * scoring.progress.done) / scoring.progress.total))}%)`
        : ""}
    </span>
  );
}

function PatientCard({
  row,
  scores,
  scoring,
  onOpen,
}: {
  row: PatientHit & { id: string };
  scores: PatientHit["scores"] | undefined;
  scoring: boolean;
  onOpen: (tab: "pair" | "summary") => void;
}) {
  const p = row.patient;
  return (
    <li className="flex gap-4 rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h3 className="text-base leading-snug font-semibold text-gray-900">
            {patientLine(p) || p.id}
          </h3>
          <MatchBadge scores={scores} scoring={scoring} />
        </div>
        <p className="flex flex-wrap gap-x-4 text-xs text-gray-500">
          <span className="font-mono">{p.id}</span>
          <span className="tabular-nums" title="TrialSpace cosine similarity">
            similarity {row.cosine.toFixed(3)}
          </span>
        </p>
        <Requirements match={scores?.match} />
        <ExclusionLine boilerplate={scores?.boilerplate} scoring={scoring} />
      </div>
      <div className="flex shrink-0 flex-col gap-2">
        <Button
          size="sm"
          onClick={() => {
            onOpen("summary");
          }}
        >
          Patient Details
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            onOpen("pair");
          }}
        >
          Check criteria
        </Button>
      </div>
    </li>
  );
}

/** The right panel: the cohort searched for, its criteria, collapsible as mm-dash's patient panel. */
function CohortPanel({ cohort }: { cohort: CohortDetail }) {
  const panel = useRef<PanelImperativeHandle>(null);
  const [hidden, setHidden] = useState(false);
  return (
    <Fragment>
      <ResizableHandle className={hidden ? "hidden" : "hover:bg-gray-400"} />
      <ResizablePanel
        id="cohort"
        panelRef={panel}
        defaultSize="38%"
        minSize="20%"
        maxSize="60%"
        collapsible
        className="bg-slate-50"
        style={{ overflow: "visible" }}
        onResize={(size) => {
          setHidden(size.asPercentage === 0);
        }}
      >
        {hidden ? null : (
          <aside className="sticky top-0 flex max-h-svh flex-col gap-3 overflow-y-auto p-4">
            <div className="flex items-center gap-2">
              <h2 className="text-xs font-semibold tracking-wider text-gray-500 uppercase">
                Searched for
              </h2>
              <span className="text-xs text-gray-600">
                {cohort.trial.protocol_no} · {VERSION_LABELS[cohort.version]}
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Hide the cohort"
                className="ml-auto text-gray-500"
                onClick={() => {
                  panel.current?.collapse();
                }}
              >
                <PanelRightCloseIcon />
              </Button>
            </div>
            <div>
              <p className="font-semibold text-gray-900">{cohort.name}</p>
              <p className="text-sm text-gray-600">{cohort.trial.short_title}</p>
              <p className="text-xs text-gray-500">
                {[cohort.trial.phase, cohort.trial.nct_id, cohort.trial.site_pi]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            </div>
            <CriteriaList items={cohort.items} />
          </aside>
        )}
      </ResizablePanel>
      {hidden ? (
        <Button
          variant="outline"
          size="sm"
          className="fixed top-20 right-4 z-10"
          onClick={() => {
            panel.current?.expand();
          }}
        >
          <PanelRightOpenIcon />
          Show cohort
        </Button>
      ) : null}
    </Fragment>
  );
}
