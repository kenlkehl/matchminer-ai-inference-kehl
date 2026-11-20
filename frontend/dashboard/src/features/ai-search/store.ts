import { useSyncExternalStore } from "react";

import type { TrialsForPatientResult } from "@/lib/api";

/** The AI tab's filters, as mm-dash's (MAIA's), narrowing the retrieved trials on the client. */
export interface AiFilters {
  phases: string[];
  centers: string[];
  locations: string[];
  /** Leave out the groups that need a higher match threshold (the site config's higher_threshold_groups). */
  excludeHigherThreshold: boolean;
  /** Leave out the trials whose common-exclusion check found the patient likely excluded. */
  hideLikelyExcluded: boolean;
}

/** The trials-for-a-patient view's URL state: its open dialogs (the results themselves stay in memory). */
export interface AiSearch {
  /** The trial whose Drugs / mechanism dialog is open. */
  drugs?: string;
  /** The trial whose Good option check dialog is open. */
  good?: string;
  /** The catalog evidence dialog open over it: `drug:<id>` or `class:<id>`. */
  evidence?: string;
}

export function validateAiSearch(search: Record<string, unknown>): AiSearch {
  const text = (v: unknown) => (typeof v === "string" && v ? v : undefined);
  return { drugs: text(search.drugs), good: text(search.good), evidence: text(search.evidence) };
}

export const DEFAULT_FILTERS: AiFilters = {
  phases: [],
  centers: [],
  locations: [],
  excludeHigherThreshold: true,
  hideLikelyExcluded: false,
};

export const CANCER_SUMMARY_SCAFFOLD = `Age:
Sex:
Cancer type:
Histology:
Current extent:
Biomarkers:
Treatment history:
`;

/**
 * What the AI tab holds while other tabs show: the patient chosen (a synthetic patient, or an entered
 * summary), the results and the filters. It stays in memory only; a reload starts over.
 */
export interface AiState {
  page: "search" | "results";
  patientId: string | null;
  summary: string;
  boilerplate: string;
  result: TrialsForPatientResult | null;
  allCohorts: boolean;
  filters: AiFilters;
  searches: number;
  /**
   * The results' patient panel height chosen with its splitter, in percent of the window height;
   * null is the default.
   */
  summaryHeight: number | null;
}

let state: AiState = {
  page: "search",
  patientId: null,
  summary: CANCER_SUMMARY_SCAFFOLD,
  boilerplate: "",
  result: null,
  allCohorts: false,
  filters: DEFAULT_FILTERS,
  searches: 0,
  summaryHeight: null,
};
const listeners = new Set<() => void>();

export function setAiState(patch: Partial<AiState> | ((s: AiState) => Partial<AiState>)) {
  state = { ...state, ...(typeof patch === "function" ? patch(state) : patch) };
  for (const listener of listeners) listener();
}

export function useAiState(): AiState {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => state,
  );
}
