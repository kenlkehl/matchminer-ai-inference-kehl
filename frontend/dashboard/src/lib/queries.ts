import { queryOptions } from "@tanstack/react-query";

import {
  api,
  type CatalogEntry,
  type CatalogKind,
  type LLMStatus,
  type CenterView,
  type CohortDetail,
  type Meta,
  type Note,
  type PatientDetail,
  type PatientList,
  type TrialDetail,
} from "./api";

/** The store's summary; polled so a dashboard picks up newly extracted trials. */
export const metaOptions = () =>
  queryOptions({
    queryKey: ["meta"],
    queryFn: () => api.get<Meta>("/api/meta"),
    refetchInterval: 30_000,
  });

export const centerOptions = (centerId: string, builtAt: number | undefined) =>
  queryOptions({
    // Keyed by the store's build time, so a rebuilt store refetches the dashboard.
    queryKey: ["center", centerId, builtAt],
    queryFn: () => api.get<CenterView>(`/api/centers/${encodeURIComponent(centerId)}`),
    staleTime: Infinity,
    placeholderData: (previous) => previous,
  });

export const trialOptions = (trialId: string) =>
  queryOptions({
    queryKey: ["trial", trialId],
    queryFn: () => api.get<TrialDetail>(`/api/trials/${encodeURIComponent(trialId)}`),
  });

/** A drug's or class's GoodOption catalog evidence (read on demand by the server). */
export const catalogOptions = (kind: CatalogKind, id: string) =>
  queryOptions({
    queryKey: ["catalog", kind, id],
    queryFn: () =>
      api.get<CatalogEntry>(
        `/api/${kind === "drug" ? "drugs" : "classes"}/${encodeURIComponent(id)}`,
      ),
    staleTime: 5 * 60_000,
  });

/** The LLM checks' endpoint and what is available with it. */
export const llmOptions = () =>
  queryOptions({
    queryKey: ["llm"],
    queryFn: () => api.get<LLMStatus>("/api/llm"),
    staleTime: 30_000,
  });

export const cohortOptions = (cohortId: string) =>
  queryOptions({
    queryKey: ["cohort", cohortId],
    queryFn: () => api.get<CohortDetail>(`/api/cohorts/${encodeURIComponent(cohortId)}`),
  });

export const patientOptions = (patientId: string) =>
  queryOptions({
    queryKey: ["patient", patientId],
    queryFn: () => api.get<PatientDetail>(`/api/patients/${encodeURIComponent(patientId)}`),
    staleTime: Infinity,
  });

export const notesOptions = (patientId: string) =>
  queryOptions({
    queryKey: ["notes", patientId],
    queryFn: () =>
      api.get<{ state: string; notes: Note[]; message?: string }>(
        `/api/patients/${encodeURIComponent(patientId)}/notes`,
      ),
    refetchInterval: (query) => (query.state.data?.state === "ready" ? false : 3_000),
  });

export const patientsOptions = (q: string, cancerType: string, sex: string, offset: number) =>
  queryOptions({
    queryKey: ["patients", q, cancerType, sex, offset],
    queryFn: () =>
      api.get<PatientList>(
        `/api/patients?${new URLSearchParams({ q, cancer_type: cancerType, sex, offset: String(offset), limit: "25" }).toString()}`,
      ),
    placeholderData: (previous) => previous,
  });
