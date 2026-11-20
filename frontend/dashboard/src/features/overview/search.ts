import type { CohortRow, DrugBrief, Version } from "@/lib/api";

/** The overview's URL state: the center, its filters, columns and open dialogs. */
export interface OverviewSearch {
  center?: string;
  q?: string;
  groups?: string[];
  phases?: string[];
  sources?: Version[];
  types?: string[];
  /** Drug mechanism classes (GoodOption catalog). */
  mechanisms?: string[];
  /** Columns whose visibility differs from the default. */
  cols?: Partial<Record<ColumnKey, boolean>>;
  /** The trial whose details dialog is open. */
  details?: string;
  /** The cohort whose criteria dialog is open. */
  cohort?: string;
  /** The GoodOption catalog evidence dialog open: `drug:<id>` or `class:<id>` (over trial details, if open). */
  evidence?: string;
}

function strings(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((v) => typeof v === "string") ? value : undefined;
}

export function validateOverviewSearch(search: Record<string, unknown>): OverviewSearch {
  const text = (v: unknown) => (typeof v === "string" && v ? v : undefined);
  const cols =
    search.cols && typeof search.cols === "object"
      ? (search.cols as OverviewSearch["cols"])
      : undefined;
  return {
    center: text(search.center),
    q: text(search.q),
    groups: strings(search.groups),
    phases: strings(search.phases),
    sources: strings(search.sources) as Version[] | undefined,
    types: strings(search.types),
    mechanisms: strings(search.mechanisms),
    cols,
    details: text(search.details),
    cohort: text(search.cohort),
    evidence: text(search.evidence),
  };
}

export type ColumnKey =
  "irbPi" | "cohort" | "eligibility" | "drugMechanism" | "contacts" | "patients";

export const COLUMNS: { key: ColumnKey; label: string; width: number; default: boolean }[] = [
  { key: "irbPi", label: "IRB# / PI", width: 1.05, default: true },
  { key: "cohort", label: "Cohort", width: 1.15, default: true },
  { key: "eligibility", label: "Key Eligibility Criteria", width: 2.5, default: true },
  { key: "drugMechanism", label: "Drug / Mechanism", width: 1.1, default: true },
  { key: "contacts", label: "Contacts", width: 1, default: true },
  { key: "patients", label: "Patients", width: 0.75, default: true },
];

export function visibleColumns(search: OverviewSearch): Record<ColumnKey, boolean> {
  return Object.fromEntries(
    COLUMNS.map((c) => [c.key, search.cols?.[c.key] ?? c.default]),
  ) as Record<ColumnKey, boolean>;
}

export interface Filters {
  q: string;
  groups: string[];
  phases: string[];
  sources: Version[];
  types: string[];
  mechanisms: string[];
}

export function filtersOf(search: OverviewSearch): Filters {
  return {
    q: search.q ?? "",
    groups: search.groups ?? [],
    phases: search.phases ?? [],
    sources: search.sources ?? [],
    types: search.types ?? [],
    mechanisms: search.mechanisms ?? [],
  };
}

export function hasFilters(f: Filters): boolean {
  return Boolean(
    f.q ||
    f.groups.length ||
    f.phases.length ||
    f.sources.length ||
    f.types.length ||
    f.mechanisms.length,
  );
}

/** The search box's terms: words, or "quoted phrases"; every term must match (case-insensitive). */
export function terms(q: string): string[] {
  return [...q.matchAll(/"([^"]*)"?|(\S+)/g)]
    .map((m) => (m[1] ?? m[2] ?? "").toLowerCase())
    .filter(Boolean);
}

/** The trial's drugs in a role: investigational and uncertain drugs are the ones studied. */
export function drugsIn(row: CohortRow, roles: readonly string[]): DrugBrief[] {
  return (row.trial.drugs ?? []).filter((d) => roles.includes(d.role));
}

export const STUDIED_ROLES = ["investigational", "uncertain"] as const;
/** The mechanism filter's roles: the drugs studied and their comparators, not background therapy. */
export const FILTER_ROLES = ["investigational", "uncertain", "control"] as const;

/** Every class (specific and generic) of the trial's studied and control drugs: the mechanism filter's values. */
export function mechanismClasses(row: CohortRow): string[] {
  return [...new Set(drugsIn(row, FILTER_ROLES).flatMap((d) => d.all_classes))];
}

export function rowText(row: CohortRow): string {
  const t = row.trial;
  return [
    t.protocol_no,
    t.nct_id,
    t.short_title,
    t.site_pi,
    t.overall_pi,
    t.contact.name,
    row.name,
    row.group ?? "",
    ...row.key_inclusion,
    ...row.key_exclusion,
    ...(t.drugs ?? []).flatMap((d) => [
      d.name,
      d.listed_name ?? "",
      d.payload ?? "",
      ...d.all_classes,
    ]),
  ]
    .join("\n")
    .toLowerCase();
}

/** Whether a row passes the filters; `except` leaves one dimension out (for that filter's options). */
export function keeps(
  row: CohortRow,
  f: Filters,
  words: string[],
  text: string,
  except?: keyof Filters,
): boolean {
  const any = (chosen: string[], value: string) => chosen.length === 0 || chosen.includes(value);
  return (
    (except === "groups" || any(f.groups, row.group ?? "")) &&
    (except === "phases" || any(f.phases, row.trial.phase)) &&
    (except === "sources" || any(f.sources, row.version)) &&
    (except === "types" ||
      f.types.length === 0 ||
      row.cancer_types.some((t) => f.types.includes(t))) &&
    (except === "mechanisms" ||
      f.mechanisms.length === 0 ||
      mechanismClasses(row).some((c) => f.mechanisms.includes(c))) &&
    (except === "q" || words.every((w) => text.includes(w)))
  );
}
