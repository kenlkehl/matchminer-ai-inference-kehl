import { cn } from "cn";
import { useState } from "react";

import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { SidebarSlot } from "@/components/shell/sidebar-slot";
import { type CenterView, VERSION_LABELS, type Version } from "@/lib/api";

import type { Overview } from "./overview-page";
import { type Filters, mechanismClasses } from "./search";

interface Option {
  value: string;
  label: string;
  count: number;
}

/** The overview's filters, in the app sidebar (mm-dash's dark filter panel). */
export function FilterPanel({ overview, view }: { overview: Overview; view: CenterView }) {
  const { rows, filters, shown, setFilters } = overview;
  // Each filter offers the values of the rows every other filter keeps, and the values chosen.
  const options = (
    key: keyof Filters,
    values: (row: (typeof rows)[number]) => string[],
    label: (v: string) => string,
  ) => {
    const counts = new Map<string, number>();
    for (const row of rows) {
      if (!shown(row, key)) continue;
      for (const v of values(row)) counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    for (const v of filters[key] as string[]) if (!counts.has(v)) counts.set(v, 0);
    return [...counts].map(([value, count]) => ({ value, label: label(value), count }));
  };
  const groupOrder = view.groups.map((g) => g.name);
  const groups = options(
    "groups",
    (r) => [r.group ?? ""],
    (v) => v,
  ).sort((a, b) => groupOrder.indexOf(a.value) - groupOrder.indexOf(b.value));
  const phases = options(
    "phases",
    (r) => [r.trial.phase],
    (v) => v || "Not given",
  ).sort((a, b) => a.value.localeCompare(b.value));
  const sources = options(
    "sources",
    (r) => [r.version],
    (v) => VERSION_LABELS[v as Version],
  );
  const types = options(
    "types",
    (r) => r.cancer_types,
    (v) => view.cancer_type_labels[v] ?? v,
  )
    .filter((o) => o.value !== "lung_any")
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
  const mechanisms = options("mechanisms", mechanismClasses, (v) => v).sort(
    (a, b) => b.count - a.count || a.label.localeCompare(b.label),
  );

  return (
    <SidebarSlot>
      <div className="mt-2 flex flex-col gap-4 border-t border-white/20 pt-4 pb-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Filters</h2>
          <button
            type="button"
            className="text-xs text-gray-400 underline hover:text-white"
            onClick={overview.clearFilters}
          >
            Clear All
          </button>
        </div>
        <CheckList
          label="Groups"
          options={groups}
          value={filters.groups}
          onChange={(v) => {
            setFilters({ groups: v.length ? v : undefined });
          }}
        />
        <CheckList
          label="Phase"
          options={phases}
          value={filters.phases}
          onChange={(v) => {
            setFilters({ phases: v.length ? v : undefined });
          }}
        />
        <CheckList
          label="Cohorts from"
          options={sources}
          value={filters.sources}
          onChange={(v) => {
            setFilters({ sources: v.length ? (v as Version[]) : undefined });
          }}
        />
        <CheckList
          label="Cancer Type"
          options={types}
          value={filters.types}
          max={12}
          onChange={(v) => {
            setFilters({ types: v.length ? v : undefined });
          }}
        />
        <CheckList
          label="Drug Mechanism"
          options={mechanisms}
          value={filters.mechanisms}
          max={12}
          expandable
          onChange={(v) => {
            setFilters({ mechanisms: v.length ? v : undefined });
          }}
        />
      </div>
    </SidebarSlot>
  );
}

export const FILTER_LABEL_CLASS =
  "text-xs font-semibold tracking-wider text-sidebar-foreground/60 uppercase";

function CheckList({
  label,
  options,
  value,
  onChange,
  max,
  expandable = false,
}: {
  label: string;
  options: Option[];
  value: string[];
  onChange: (value: string[]) => void;
  /** Show the first `max` options (and the chosen ones). */
  max?: number;
  /** Offer the rest with a "Show all" toggle. */
  expandable?: boolean;
}) {
  const [all, setAll] = useState(false);
  if (options.length === 0) return null;
  const truncated = Boolean(max && options.length > max);
  const shown =
    max && truncated && !all
      ? options.filter((o, i) => i < max || value.includes(o.value))
      : options;
  return (
    <fieldset className="flex flex-col gap-1.5">
      <legend className={cn(FILTER_LABEL_CLASS, "mb-1.5")}>{label}</legend>
      {shown.map((o) => (
        <Label
          key={o.value}
          className="items-start gap-1.5 text-xs leading-snug font-normal text-gray-200"
        >
          <Checkbox
            className="mt-px size-3.5 shrink-0"
            checked={value.includes(o.value)}
            onCheckedChange={(checked) => {
              onChange(checked ? [...value, o.value] : value.filter((v) => v !== o.value));
            }}
          />
          <span className={cn("flex-1", o.count === 0 && "text-white/40")}>{o.label}</span>
          <span className="text-white/40 tabular-nums">{o.count}</span>
        </Label>
      ))}
      {expandable && truncated ? (
        <button
          type="button"
          className="self-start text-xs text-gray-400 underline hover:text-white"
          onClick={() => {
            setAll((v) => !v);
          }}
        >
          {all ? "Show fewer" : `Show all ${String(options.length)}`}
        </button>
      ) : null}
    </fieldset>
  );
}
