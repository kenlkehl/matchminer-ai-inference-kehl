import { useMutation, useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";

import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { api } from "@/lib/api";
import { isoWhen, when } from "@/lib/format";

import { LLMPanel } from "./llm-panel";

interface Status {
  store: {
    built_at: number;
    build_seconds: number;
    counts: Record<string, unknown>;
    rules: { path: string; centers: { match?: string; groups?: string[]; error?: string }[] };
    error: string | null;
    last_check: number | null;
    path: string;
    poll_seconds: number;
  };
  sources: Record<string, string>;
  good_options?: GoodOptions | null;
  patients: { corpus: string; count: number };
  round: { round: number; encoder: Record<string, string>; model: { model: string } } | null;
  skipped_rounds: { round: number; reason: string }[];
  encoder: { available: boolean; problem: string | null; state: string; device: string };
  cohort_vectors: { state: string; vectors: number; missing?: number; error: string | null };
  decision_model: {
    model: string;
    why: string;
    device: string;
    loaded: boolean;
    questions_sha256: string;
  };
  jobs: {
    id: string;
    kind: string;
    label: string;
    status: string;
    done: number;
    total: number;
    seconds: number | null;
    error: string | null;
  }[];
  score_cache: number;
}

interface GoodOptions {
  path: string | null;
  state: "ready" | "missing" | "error";
  message: string | null;
  build_id: string | null;
  created_at: string | null;
  ncit: { resource?: string; version?: string } | null;
  counts: Record<string, number> | null;
  most_specific: boolean;
  hierarchy: string | null;
  shown: {
    trials: number;
    trials_with_drugs: number;
    drugs: number;
    distinct_drugs: number;
    mechanism_types: number;
    filter_classes: number;
  };
}

/** The GoodOption catalog the trials' drugs come from, or why there are none. */
function GoodOptionsRow({ path, info }: { path: string | undefined; info: GoodOptions | null }) {
  if (!info) {
    return (
      <Row label="GoodOption catalog">
        {path ? `${path}: ` : ""}not in this store yet (a server running this version rebuilds the
        store with the trials' drugs)
      </Row>
    );
  }
  const c = info.counts ?? {};
  return (
    <Row label="GoodOption catalog">
      {info.path}
      {info.state === "ready" ? (
        <>
          <p className="text-xs text-gray-500">
            Built {isoWhen(info.created_at)} (build {info.build_id}); {c.trials} trials,{" "}
            {c.unique_drugs} drugs, {c.unique_classes} classes
            {info.ncit?.version ? `, NCIt ${info.ncit.version}` : ""}
          </p>
          <p className="text-xs text-gray-500">
            Shown: {info.shown.trials_with_drugs} of {info.shown.trials} trials with drugs,{" "}
            {info.shown.drugs} drug entries ({info.shown.distinct_drugs} drugs),{" "}
            {info.shown.mechanism_types} mechanism types, {info.shown.filter_classes} filter classes
          </p>
          <p className="text-xs text-gray-500">
            Mechanism types:{" "}
            {info.most_specific
              ? `each drug's most specific classes (${info.hierarchy ?? "NCIt"})`
              : info.hierarchy}
          </p>
        </>
      ) : null}
      {info.message ? (
        <p className={info.state === "ready" ? "text-xs text-gray-500" : "text-red-700"}>
          {info.message}
        </p>
      ) : null}
    </Row>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[12rem_1fr] gap-3 py-1.5 text-sm">
      <dt className="text-gray-500">{label}</dt>
      <dd className="break-words text-gray-800">{children}</dd>
    </div>
  );
}

/** Where the dashboard's data come from and which models it runs. */
export function StatusPage() {
  const status = useQuery({
    queryKey: ["status"],
    queryFn: () => api.get<Status>("/api/status"),
    refetchInterval: 5_000,
  });
  const rebuild = useMutation({ mutationFn: () => api.post("/api/rebuild", {}) });
  const s = status.data;
  if (!s) return <Spinner className="size-6" />;
  return (
    <div className="flex flex-col gap-6">
      <PageHeader title="Data and Models" />
      <section>
        <h2 className="mb-2 text-lg font-semibold">Trials</h2>
        <dl className="divide-y divide-gray-100">
          <Row label="Built">
            {when(s.store.built_at)} ({s.store.build_seconds} s); checked for new extractions every{" "}
            {s.store.poll_seconds} s{s.store.last_check ? `, last ${when(s.store.last_check)}` : ""}{" "}
            <Button
              size="xs"
              variant="outline"
              className="ml-2"
              disabled={rebuild.isPending}
              onClick={() => {
                rebuild.mutate();
              }}
            >
              Check now
            </Button>
          </Row>
          {s.store.error ? <Row label="Last rebuild error">{s.store.error}</Row> : null}
          <Row label="Counts">
            <code className="text-xs">{JSON.stringify(s.store.counts)}</code>
          </Row>
          <Row label="Trial data">{s.sources.trial_data}</Row>
          <Row label="OA database">{s.sources.oa_db}</Row>
          <Row label="Grouping rules">
            {s.sources.grouping}
            {s.store.rules.centers
              .filter((c) => c.error)
              .map((c) => (
                <p key={c.error} className="text-red-700">
                  {c.error}
                </p>
              ))}
          </Row>
          <GoodOptionsRow path={s.sources.good_options} info={s.good_options ?? null} />
          <Row label="Store">{s.store.path}</Row>
        </dl>
      </section>
      <section>
        <h2 className="mb-2 text-lg font-semibold">Matching</h2>
        <dl className="divide-y divide-gray-100">
          <Row label="Synthetic patients">
            {s.patients.count.toLocaleString()} ({s.patients.corpus})
          </Row>
          <Row label="TrialSpace round">
            {s.round ? `round ${String(s.round.round)} · ${s.round.model.model}` : "none"}
            {s.skipped_rounds.map((r) => (
              <p key={r.round} className="text-xs text-gray-500">
                round {r.round} skipped: {r.reason}
              </p>
            ))}
          </Row>
          <Row label="Encoder">
            {s.encoder.available ? `${s.encoder.state} on ${s.encoder.device}` : s.encoder.problem}
          </Row>
          <Row label="Trial cohort vectors">
            {s.cohort_vectors.state}: {s.cohort_vectors.vectors} embedded
            {s.cohort_vectors.missing ? `, ${String(s.cohort_vectors.missing)} to go` : ""}
            {s.cohort_vectors.error ? ` (${s.cohort_vectors.error})` : ""}
          </Row>
          <Row label="Decision model">
            {s.decision_model.model} on {s.decision_model.device} (
            {s.decision_model.loaded ? "loaded" : "loading"}) — {s.decision_model.why}
          </Row>
          <Row label="Cached answers">{s.score_cache.toLocaleString()}</Row>
        </dl>
      </section>
      <section>
        <h2 className="mb-2 text-lg font-semibold">LLM checks</h2>
        <LLMPanel />
      </section>
      <section>
        <h2 className="mb-2 text-lg font-semibold">Recent jobs</h2>
        <ul className="flex flex-col gap-1 text-sm">
          {s.jobs.map((j) => (
            <li key={j.id} className="text-gray-700">
              <strong>{j.kind}</strong> {j.label}: {j.status}{" "}
              {j.total ? `${String(j.done)}/${String(j.total)}` : ""}
              {j.seconds ? ` · ${String(j.seconds)} s` : ""}
              {j.error ? <span className="text-red-700"> {j.error}</span> : null}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
