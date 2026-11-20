import { cn } from "cn";
import { Fragment, type ReactNode } from "react";

import type { CatalogKind, DrugBrief } from "@/lib/api";

/** Opens a drug's or class's GoodOption catalog evidence: `drug:<id>` or `class:<id>`. */
export type OpenEvidence = (ref: string) => void;

export function evidenceRef(kind: CatalogKind, id: string): string {
  return `${kind}:${id}`;
}

export function parseEvidenceRef(ref: string): { kind: CatalogKind; id: string } | null {
  const at = ref.indexOf(":");
  const kind = ref.slice(0, at);
  if ((kind !== "drug" && kind !== "class") || ref.length <= at + 1) return null;
  return { kind, id: ref.slice(at + 1) };
}

const STYLES = {
  /** A drug's name: a link. */
  link: "link inline",
  /** A class name inside text: dotted underline, blue on hover. */
  subtle:
    "inline underline decoration-gray-400 decoration-dotted underline-offset-2 hover:text-brand-blue hover:decoration-brand-blue",
  /** A chip (its own border and padding): blue on hover. */
  chip: "hover:border-brand-blue hover:text-brand-blue",
};

/**
 * A drug or class name that opens its evidence dialog (a button, so it is keyboard-accessible; the
 * click goes no further than the dialog). Plain text when there is no ID (a store older than the
 * dialogs) or no handler.
 */
export function CatalogLink({
  kind,
  id,
  onOpen,
  variant = "subtle",
  className,
  children,
}: {
  kind: CatalogKind;
  id: string | undefined;
  onOpen: OpenEvidence | undefined;
  variant?: keyof typeof STYLES;
  className?: string;
  children: ReactNode;
}) {
  if (!id || !onOpen) return <span className={className}>{children}</span>;
  return (
    <button
      type="button"
      className={cn(
        "cursor-pointer text-left focus-visible:rounded-xs focus-visible:outline-2 focus-visible:outline-brand-blue",
        STYLES[variant],
        className,
      )}
      title={`The GoodOption catalog's evidence for this ${kind === "drug" ? "drug" : "class"}`}
      onClick={(event) => {
        event.stopPropagation();
        onOpen(evidenceRef(kind, id));
      }}
    >
      {children}
    </button>
  );
}

/** Class names, comma-separated, each opening its evidence. */
export function ClassLinks({
  names,
  ids,
  onOpen,
}: {
  names: string[];
  ids: Record<string, string> | undefined;
  onOpen: OpenEvidence | undefined;
}) {
  return names.map((name, i) => (
    <Fragment key={name}>
      {i > 0 ? ", " : null}
      <CatalogLink kind="class" id={ids?.[name]} onOpen={onOpen}>
        {name}
      </CatalogLink>
    </Fragment>
  ));
}

/** An ADC's payload: its payload classes as links ("Exatecan Derivative (Topoisomerase I Inhibitor)"),
 * or the plain text the catalog's facts gave. */
export function PayloadLinks({
  drug,
  onOpen,
}: {
  drug: DrugBrief;
  onOpen: OpenEvidence | undefined;
}): ReactNode {
  const parts = drug.payload_classes ?? [];
  if (parts.length === 0) return drug.payload;
  return parts.map((part, i) => (
    <Fragment key={part.class_name}>
      {i > 0 ? ", " : null}
      <CatalogLink kind="class" id={drug.class_ids?.[part.class_name]} onOpen={onOpen}>
        {part.label}
      </CatalogLink>
      {part.generic.length > 0 ? (
        <>
          {" ("}
          {part.generic.map((g, j) => (
            <Fragment key={g.class_name}>
              {j > 0 ? ", " : null}
              <CatalogLink kind="class" id={drug.class_ids?.[g.class_name]} onOpen={onOpen}>
                {g.label}
              </CatalogLink>
            </Fragment>
          ))}
          {")"}
        </>
      ) : null}
    </Fragment>
  ));
}

/** Drug names, comma-separated, each opening its evidence. */
export function DrugLinks({
  drugs,
  onOpen,
}: {
  drugs: DrugBrief[];
  onOpen: OpenEvidence | undefined;
}) {
  return drugs.map((d, i) => (
    <Fragment key={d.drug_id}>
      {i > 0 ? ", " : null}
      <CatalogLink kind="drug" id={d.drug_id} onOpen={onOpen}>
        {d.name}
      </CatalogLink>
    </Fragment>
  ));
}
