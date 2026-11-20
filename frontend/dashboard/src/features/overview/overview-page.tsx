import { useQuery } from "@tanstack/react-query";
import { getRouteApi, Link } from "@tanstack/react-router";
import { ChevronsDownUpIcon, ChevronsUpDownIcon, Columns3Icon } from "lucide-react";
import { useMemo, useState } from "react";

import { PageHeader } from "@/components/page-header";
import { SearchInput } from "@/components/search-input";
import { Accordion } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Field, FieldLabel } from "@/components/ui/field";
import { Spinner } from "@/components/ui/spinner";
import type { CenterInfo, CenterView, CohortRow } from "@/lib/api";
import { when } from "@/lib/format";
import { centerOptions, metaOptions } from "@/lib/queries";

import { CatalogEvidenceDialog } from "./catalog-evidence-dialog";
import { CohortDialog } from "./cohort-dialog";
import { FilterPanel } from "./filter-panel";
import {
  COLUMNS,
  filtersOf,
  hasFilters,
  keeps,
  type OverviewSearch,
  rowText,
  terms,
  visibleColumns,
} from "./search";
import { PendingSection, TrialGroupSection } from "./trial-group-section";
import { TrialDetailsDialog } from "./trial-details-dialog";

const route = getRouteApi("/_app/");

export function OverviewPage() {
  const search = route.useSearch();
  const meta = useQuery(metaOptions());
  if (!meta.data) {
    return meta.isError ? (
      <p className="text-sm text-red-700">Could not load the dashboard: {meta.error.message}</p>
    ) : (
      <Loading />
    );
  }
  const center = meta.data.centers.find((c) => c.id === search.center);
  return center ? (
    <CenterOverview center={center} builtAt={meta.data.built_at} />
  ) : (
    <Landing
      centers={meta.data.centers}
      builtAt={meta.data.built_at}
      trials={meta.data.counts.trials}
    />
  );
}

function Loading() {
  return (
    <div className="flex justify-center py-16">
      <Spinner className="size-6" />
    </div>
  );
}

/** No center chosen: every disease center's dashboard, largest first. */
function Landing({
  centers,
  builtAt,
  trials,
}: {
  centers: CenterInfo[];
  builtAt: number;
  trials: number;
}) {
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Trials Overview">
        <p>
          {trials} trials across {centers.length} disease centers, from their protocols (or their
          ClinicalTrials.gov listings) as extracted so far; updated {when(builtAt)}. Choose a
          disease center to see its trials and cohorts.
        </p>
      </PageHeader>
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-3">
        {centers.map((c) => (
          <li key={c.id}>
            <Link
              to="/"
              search={{ center: c.id }}
              className="flex h-full flex-col gap-1 rounded-lg border border-gray-200 bg-white p-4 hover:border-brand-blue hover:bg-brand-blue/5"
            >
              <span className="font-semibold text-gray-900">{c.short_name}</span>
              <span className="text-xs text-gray-500">{c.name}</span>
              <span className="mt-auto pt-2 text-sm text-gray-700">
                {c.trials} trials · {c.cohorts} cohorts
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export type Overview = ReturnType<typeof useOverview>;

function useOverview(view: CenterView | undefined) {
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const filters = filtersOf(search);
  const [clears, setClears] = useState(0);
  const go = (update: (prev: OverviewSearch) => OverviewSearch) => {
    void navigate({ search: update, resetScroll: false });
  };
  const rows = useMemo(() => (view ? view.groups.flatMap((g) => g.rows) : []), [view]);
  const texts = useMemo(() => new Map(rows.map((r) => [r.cohort_id, rowText(r)])), [rows]);
  const words = terms(filters.q);
  const shown = (row: CohortRow, except?: keyof typeof filters) =>
    keeps(row, filters, words, texts.get(row.cohort_id) ?? "", except);
  return {
    search,
    filters,
    rows,
    shown,
    clears,
    columns: visibleColumns(search),
    setFilters: (patch: Partial<OverviewSearch>) => {
      go((prev) => ({ ...prev, ...patch }));
    },
    clearFilters: () => {
      setClears((n) => n + 1);
      go((prev) => ({ center: prev.center, cols: prev.cols }));
    },
    setColumn: (key: keyof ReturnType<typeof visibleColumns>, visible: boolean) => {
      go((prev) => ({ ...prev, cols: { ...prev.cols, [key]: visible } }));
    },
    open: (patch: Pick<OverviewSearch, "details" | "cohort">) => {
      go((prev) => ({
        ...prev,
        details: undefined,
        cohort: undefined,
        evidence: undefined,
        ...patch,
      }));
    },
    /** A drug's or class's catalog evidence (`drug:<id>`, `class:<id>`), over trial details if open. */
    showEvidence: (evidence: string | undefined) => {
      go((prev) => ({ ...prev, evidence }));
    },
  };
}

function CenterOverview({ center, builtAt }: { center: CenterInfo; builtAt: number }) {
  const query = useQuery(centerOptions(center.id, builtAt));
  const view = query.data?.id === center.id ? query.data : undefined;
  const overview = useOverview(view);
  const [openGroups, setOpenGroups] = useState<string[] | null>(null);
  const { filters, search } = overview;
  const filtering = hasFilters(filters);

  const groups = (view?.groups ?? [])
    .map((g) => ({ ...g, rows: g.rows.filter((r) => overview.shown(r)) }))
    .filter((g) => g.rows.length > 0);
  const pending = filtering ? [] : (view?.pending ?? []);
  // While filtering, the groups with matches open; otherwise as the user left them (all closed at first).
  const open = openGroups ?? (filtering ? groups.map((g) => g.id) : []);
  const allIds = [...groups.map((g) => g.id), ...(pending.length ? ["pending"] : [])];
  const trialCount = new Set(groups.flatMap((g) => g.rows.map((r) => r.trial.id))).size;
  // The catalog evidence dialog: nested in trial details when those are open, so it stacks over them.
  const evidence = search.evidence ? (
    <CatalogEvidenceDialog
      target={search.evidence}
      onClose={() => {
        overview.showEvidence(undefined);
      }}
      onOpen={overview.showEvidence}
      onTrial={(details) => {
        overview.open({ details });
      }}
    />
  ) : null;
  const cohortCount = groups.reduce((n, g) => n + g.rows.length, 0);

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title={`${center.short_name} Trials Overview`}>
        <p>
          {center.name}: {center.trials} trials, with each cohort's eligibility criteria as
          extracted from its protocol (or, before its protocol is processed, its ClinicalTrials.gov
          listing). Updated {when(builtAt)}; new extractions appear here as they finish.
        </p>
      </PageHeader>
      {view ? <FilterPanel overview={overview} view={view} /> : null}
      <div className="flex items-end gap-2 print:hidden">
        <Field className="flex-1">
          <FieldLabel htmlFor="overview-search" className="text-sm font-normal text-gray-600">
            Search by Protocol, PI, Title, Cohort, Criteria, Drug
          </FieldLabel>
          <SearchInput
            key={`${center.id}-${String(overview.clears)}`}
            id="overview-search"
            placeholder="Type to search..."
            value={filters.q}
            onChange={(q) => {
              setOpenGroups(null);
              overview.setFilters({ q: q || undefined });
            }}
          />
        </Field>
        <Button
          variant="outline"
          className="font-normal"
          onClick={() => {
            setOpenGroups(open.length < allIds.length ? allIds : []);
          }}
        >
          {open.length < allIds.length ? <ChevronsUpDownIcon /> : <ChevronsDownUpIcon />}
          {open.length < allIds.length ? "Expand all" : "Collapse all"}
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="outline" className="font-normal" />}>
            <Columns3Icon />
            Columns
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuGroup>
              <DropdownMenuLabel>Show / hide columns</DropdownMenuLabel>
              {COLUMNS.map((column) => (
                <DropdownMenuCheckboxItem
                  key={column.key}
                  checked={overview.columns[column.key]}
                  onCheckedChange={(checked) => {
                    overview.setColumn(column.key, checked);
                  }}
                >
                  {column.label}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      {!view ? (
        query.isError ? (
          <p className="text-sm text-red-700">
            Could not load {center.short_name}: {query.error.message}
          </p>
        ) : (
          <Loading />
        )
      ) : (
        <>
          <Accordion
            multiple
            value={open}
            onValueChange={(value) => {
              setOpenGroups(value as string[]);
            }}
            className="overflow-hidden rounded-md border border-gray-300"
          >
            {groups.map((group) => (
              <TrialGroupSection
                key={group.id}
                group={group}
                columns={overview.columns}
                labels={view.cancer_type_labels}
                onDetails={(details) => {
                  overview.open({ details });
                }}
                onCriteria={(cohort) => {
                  overview.open({ cohort });
                }}
                onEvidence={overview.showEvidence}
              />
            ))}
            {pending.length ? (
              <PendingSection
                trials={pending}
                onDetails={(details) => {
                  overview.open({ details });
                }}
              />
            ) : null}
          </Accordion>
          <p className="text-sm text-gray-500">
            {cohortCount > 0
              ? `Showing ${String(trialCount)} trials, ${String(cohortCount)} cohorts across ${String(groups.length)} groups`
              : "No trials found."}
            {pending.length ? ` · ${String(pending.length)} trials without extracted cohorts` : ""}
          </p>
        </>
      )}
      {search.details ? (
        <TrialDetailsDialog
          trialId={search.details}
          onClose={() => {
            overview.open({});
          }}
          onCriteria={(cohort) => {
            overview.open({ cohort });
          }}
          onEvidence={overview.showEvidence}
        >
          {evidence}
        </TrialDetailsDialog>
      ) : (
        evidence
      )}
      {search.cohort ? (
        <CohortDialog
          cohortId={search.cohort}
          onClose={() => {
            overview.open({});
          }}
          onDetails={(details) => {
            overview.open({ details });
          }}
        />
      ) : null}
    </div>
  );
}
