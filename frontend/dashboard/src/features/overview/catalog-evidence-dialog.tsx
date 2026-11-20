import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { ExternalLinkIcon } from "lucide-react";
import { type ReactNode, useState } from "react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import type { CatalogEntry, CatalogFact, CatalogPassage } from "@/lib/api";
import { isoWhen } from "@/lib/format";
import { catalogOptions } from "@/lib/queries";

import { CatalogLink, type OpenEvidence, parseEvidenceRef } from "./catalog-links";

const SECTION_HEADING = "mb-1 text-sm font-bold";
const CHIP =
  "inline-flex max-w-full items-center gap-1 rounded-full border px-2 py-0.5 text-xs break-all";
const ROLE_LABELS: Record<string, string> = {
  investigational: "investigational",
  uncertain: "role uncertain",
  control: "control",
  background: "background",
};
const SHOWN = 30;

/**
 * A drug's or class's GoodOption catalog evidence (`?evidence=drug:<id>` or `class:<id>`): its facts by
 * facet, each citing its sources (expand a citation for its passage), its classes or drugs, and the
 * full catalog summary. Rendered inside trial details when those are open, so it stacks over them.
 */
export function CatalogEvidenceDialog({
  target,
  onClose,
  onOpen,
  onTrial,
}: {
  target: string;
  onClose: () => void;
  onOpen: OpenEvidence;
  onTrial: (trialId: string) => void;
}) {
  const ref = parseEvidenceRef(target);
  const entry = useQuery({
    ...catalogOptions(ref?.kind ?? "drug", ref?.id ?? ""),
    enabled: ref !== null,
  });
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-3xl" bodyClassName="gap-4">
        {ref === null ? (
          <DialogTitle>Not a catalog drug or class: {target}</DialogTitle>
        ) : entry.data ? (
          <EntryView key={target} entry={entry.data} onOpen={onOpen} onTrial={onTrial} />
        ) : entry.isError ? (
          <DialogTitle>Could not load the catalog's evidence: {entry.error.message}</DialogTitle>
        ) : (
          <>
            <DialogTitle className="sr-only">Loading the catalog's evidence</DialogTitle>
            <div className="flex justify-center py-12">
              <Spinner className="size-6" />
            </div>
          </>
        )}
        <DialogFooter className="pt-1">
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function EntryView({
  entry,
  onOpen,
  onTrial,
}: {
  entry: CatalogEntry;
  onOpen: OpenEvidence;
  onTrial: (trialId: string) => void;
}) {
  const c = entry.counts;
  return (
    <>
      <DialogHeader className="gap-2">
        <DialogTitle>{entry.name}</DialogTitle>
        <DialogDescription className="text-gray-600">
          {entry.kind === "drug" ? "Drug" : "Drug class"}
          {entry.ncit_code ? ` · NCIt ${entry.ncit_code}` : ""} · {c.facts} facts citing{" "}
          {c.cited_sources} sources ({c.resolved_sources} in the catalog)
        </DialogDescription>
        <p className="rounded-md bg-amber-50 px-3 py-1.5 text-xs text-amber-900">
          {entry.label}
          {entry.catalog.created_at ? `; built ${isoWhen(entry.catalog.created_at)}` : ""}
        </p>
      </DialogHeader>
      {entry.classes?.length ? (
        <section>
          <h3 className={SECTION_HEADING}>Classes</h3>
          <ul className="flex flex-wrap gap-1.5">
            {entry.classes.map((k) => (
              <li key={k.id || k.name}>
                <CatalogLink
                  kind="class"
                  id={k.id}
                  onOpen={onOpen}
                  variant="chip"
                  className={cn(
                    CHIP,
                    k.specific ? "border-gray-400 font-semibold" : "border-gray-200",
                  )}
                >
                  {k.name}
                </CatalogLink>
              </li>
            ))}
          </ul>
          {entry.classes.some((k) => k.specific) ? (
            <p className="mt-1 text-xs text-gray-500">
              In bold: its most specific classes, shown as its Type.
            </p>
          ) : null}
        </section>
      ) : null}
      {entry.drugs?.length ? (
        <section>
          <h3 className={SECTION_HEADING}>Drugs in this class ({entry.drugs.length})</h3>
          <Capped
            items={entry.drugs}
            render={(d) => (
              <li key={d.id}>
                <CatalogLink
                  kind="drug"
                  id={d.id}
                  onOpen={onOpen}
                  variant="chip"
                  className={cn(CHIP, "border-gray-200")}
                >
                  {d.name}
                </CatalogLink>
              </li>
            )}
            className="flex flex-wrap gap-1.5"
          />
        </section>
      ) : null}
      {entry.facets.length ? (
        <section>
          <h3 className={SECTION_HEADING}>Facts</h3>
          <Accordion
            multiple
            defaultValue={entry.facets.slice(0, 1).map((f) => f.facet)}
            className="overflow-hidden rounded-md border border-gray-200"
          >
            {entry.facets.map((f) => (
              <AccordionItem key={f.facet} value={f.facet} className="border-gray-200">
                <AccordionTrigger className="px-3 py-2 text-sm font-semibold text-gray-900 hover:bg-gray-50">
                  {f.label} ({f.facts.length})
                </AccordionTrigger>
                <AccordionContent className="pb-0">
                  <ul className="flex flex-col divide-y divide-gray-100">
                    {f.facts.map((fact, i) => (
                      <FactView key={i} fact={fact} evidence={entry.evidence} />
                    ))}
                  </ul>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </section>
      ) : (
        <p className="text-sm text-gray-500">The catalog has no facts for this {entry.kind}.</p>
      )}
      {entry.trials?.length ? (
        <section>
          <h3 className={SECTION_HEADING}>
            Trials here with a drug in this class ({entry.trials.length})
          </h3>
          <Capped
            items={entry.trials}
            shown={10}
            render={(t) => (
              <li key={t.id} className="text-sm">
                <button
                  type="button"
                  className="link text-left"
                  onClick={(event) => {
                    event.stopPropagation();
                    onTrial(t.id);
                  }}
                >
                  {t.protocol_no || t.id}
                </button>{" "}
                <span className="text-gray-700">{t.short_title}</span>{" "}
                <span className="text-xs text-gray-500">
                  {t.drugs.map((d) => `${d.name} (${ROLE_LABELS[d.role] ?? d.role})`).join(", ")}
                  {t.center ? ` · ${t.center}` : ""}
                </span>
              </li>
            )}
            className="flex flex-col gap-0.5"
          />
        </section>
      ) : null}
      {entry.summary ? (
        <Accordion className="overflow-hidden rounded-md border border-gray-200">
          <AccordionItem value="summary">
            <AccordionTrigger className="px-3 py-2 text-sm font-semibold text-gray-900 hover:bg-gray-50">
              Full catalog summary
            </AccordionTrigger>
            <AccordionContent>
              <p className="px-3 pb-3 text-xs whitespace-pre-wrap text-gray-700">{entry.summary}</p>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      ) : null}
    </>
  );
}

/** The first `shown` items, and a toggle for the rest. */
function Capped<T>({
  items,
  render,
  className,
  shown = SHOWN,
}: {
  items: T[];
  render: (item: T) => ReactNode;
  className: string;
  shown?: number;
}) {
  const [all, setAll] = useState(false);
  return (
    <>
      <ul className={className}>{(all ? items : items.slice(0, shown)).map(render)}</ul>
      {items.length > shown ? (
        <button
          type="button"
          className="mt-1 link text-xs"
          onClick={() => {
            setAll((v) => !v);
          }}
        >
          {all ? "Show fewer" : `Show all ${String(items.length)}`}
        </button>
      ) : null}
    </>
  );
}

function FactView({
  fact,
  evidence,
}: {
  fact: CatalogFact;
  evidence: Record<string, CatalogPassage>;
}) {
  const [open, setOpen] = useState<string[]>([]);
  return (
    <li className="flex flex-col gap-1 px-3 py-2">
      <p className="text-gray-800">{fact.text}</p>
      {fact.qualifiers.length ? (
        <p className="text-xs text-gray-500">
          {fact.qualifiers
            .map(([name, value]) => `${name.replace(/_/g, " ")}: ${value}`)
            .join(" · ")}
        </p>
      ) : null}
      {fact.support_ids.length ? (
        <div className="flex flex-wrap gap-1">
          {fact.support_ids.map((id) =>
            evidence[id] ? (
              <button
                key={id}
                type="button"
                aria-expanded={open.includes(id)}
                className={cn(
                  CHIP,
                  "cursor-pointer",
                  open.includes(id)
                    ? "border-brand-blue bg-brand-blue/5 text-brand-blue"
                    : "border-gray-300 text-gray-700 hover:border-brand-blue",
                )}
                onClick={() => {
                  setOpen((v) => (v.includes(id) ? v.filter((x) => x !== id) : [...v, id]));
                }}
              >
                {id}
              </button>
            ) : (
              <span
                key={id}
                className={cn(CHIP, "border-dashed border-gray-300 text-gray-400")}
                title="The catalog has no passage for this citation"
              >
                {id} · source not in catalog
              </span>
            ),
          )}
        </div>
      ) : null}
      {open.map((id) => (evidence[id] ? <PassageView key={id} passage={evidence[id]} /> : null))}
    </li>
  );
}

function PassageView({ passage: p }: { passage: CatalogPassage }) {
  const facts = [
    p.locator ? `Locator: ${p.locator}` : "",
    p.published_at ? `Published ${p.published_at}` : "",
    p.license ? `License: ${p.license}` : "",
  ].filter(Boolean);
  return (
    <div className="flex flex-col gap-1 rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-xs">
      <p className="text-gray-500">
        {[p.source, p.source_type].filter(Boolean).join(" · ")} · {p.evidence_id}
      </p>
      {p.title ? <p className="font-semibold text-gray-900">{p.title}</p> : null}
      <p className="max-h-60 overflow-y-auto whitespace-pre-wrap text-gray-700">{p.passage}</p>
      {p.url ? (
        <a
          className="inline-flex link items-center gap-1 self-start"
          href={p.url}
          target="_blank"
          rel="noopener noreferrer"
        >
          Open the source <ExternalLinkIcon className="size-3" />
        </a>
      ) : null}
      {facts.length ? <p className="text-gray-500">{facts.join(" · ")}</p> : null}
    </div>
  );
}
