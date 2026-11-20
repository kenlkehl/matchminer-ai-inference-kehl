import { useCallback, useEffect, useRef, useState } from "react";

import { api, type Job, type PairScores, type ScoreResult, waitForJob } from "@/lib/api";

type Direction = "patients_for_cohort" | "trials_for_patient";

export interface Scoring {
  scores: Record<string, PairScores>;
  /** The ids being scored now, and the job's progress. */
  running: string[];
  progress: { done: number; total: number; message: string } | null;
  error: string | null;
  score: (ids: string[]) => void;
}

/**
 * The decision model's answers for one search's results: what the server had cached, then the first
 * `autoN` results without a match score, scored in a background job as the page opens; more on request.
 * Jobs run one after another.
 */
export function useScoring(
  direction: Direction,
  queryId: string | null,
  results: { id: string; scores: PairScores }[] | null,
  autoN: number,
): Scoring {
  const [scores, setScores] = useState<Record<string, PairScores>>({});
  const [running, setRunning] = useState<string[]>([]);
  const [progress, setProgress] = useState<Scoring["progress"]>(null);
  const [error, setError] = useState<string | null>(null);
  const queue = useRef<string[][]>([]);
  const busy = useRef(false);
  const generation = useRef(0);

  const pump = useCallback(() => {
    if (busy.current || !queryId) return;
    const next = queue.current.shift();
    if (!next) return;
    busy.current = true;
    const mine = generation.current;
    setRunning(next);
    setProgress({ done: 0, total: next.length, message: "queued" });
    api
      .post<{ job: Job }>("/api/score", { direction, query_id: queryId, target_ids: next })
      .then(({ job }) =>
        waitForJob<ScoreResult>(job.id, (j) => {
          if (mine === generation.current)
            setProgress({ done: j.done, total: j.total, message: j.message });
        }),
      )
      .then((result) => {
        if (mine !== generation.current) return;
        setScores((current) => {
          const out = { ...current };
          for (const s of result.scores) {
            const id = direction === "patients_for_cohort" ? s.patient_id : s.cohort_id;
            out[id] = { match: s.match, boilerplate: s.boilerplate };
          }
          return out;
        });
      })
      .catch((e: unknown) => {
        if (mine === generation.current) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        busy.current = false;
        if (mine === generation.current) {
          setRunning([]);
          setProgress(null);
        }
        pump();
      });
  }, [direction, queryId]);

  // A new search: start from the server's cached answers, and score the first results lacking one.
  useEffect(() => {
    generation.current += 1;
    queue.current = [];
    busy.current = false;
    setRunning([]);
    setProgress(null);
    setError(null);
    if (!results) {
      setScores({});
      return;
    }
    setScores(Object.fromEntries(results.map((r) => [r.id, r.scores])));
    const todo = results
      .slice(0, autoN)
      .filter((r) => !r.scores.match)
      .map((r) => r.id);
    if (todo.length) {
      queue.current.push(todo);
      pump();
    }
  }, [results, autoN, pump]);

  const score = useCallback(
    (ids: string[]) => {
      const queued = new Set([...running, ...queue.current.flat()]);
      const todo = ids.filter((id) => !queued.has(id));
      if (!todo.length) return;
      queue.current.push(todo);
      pump();
    },
    [running, pump],
  );

  return { scores, running, progress, error, score };
}
