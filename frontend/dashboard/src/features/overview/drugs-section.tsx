import { Badge } from "@/components/ui/badge";
import type { Drug } from "@/lib/api";

import { CatalogLink, ClassLinks, type OpenEvidence, PayloadLinks } from "./catalog-links";

const ROLE_LABELS: Record<string, string> = {
  investigational: "Investigational",
  uncertain: "Role uncertain",
  control: "Control",
  background: "Background",
};

/** The grouped layout's groups: the drugs studied, then their comparators ("vs"), then background ("with"). */
const GROUPS: { heading: string; roles: string[] }[] = [
  { heading: "Studied", roles: ["investigational", "uncertain"] },
  { heading: "vs (control)", roles: ["control"] },
  { heading: "with (background therapy)", roles: ["background"] },
];

/**
 * A trial's drugs in the GoodOption catalog (supportive drugs left out): each one's role, name, Type,
 * Payload, mechanism sentence and arms; each drug and class name opens its catalog evidence. In role
 * order (trial details), or `grouped` under Studied / vs / with headings (the trials-for-a-patient
 * view's drugs dialog). `heading` shows the section's "Drugs" title.
 */
export function DrugsSection({
  drugs,
  onEvidence,
  grouped = false,
  heading = true,
}: {
  drugs: Drug[];
  onEvidence: OpenEvidence | undefined;
  grouped?: boolean;
  heading?: boolean;
}) {
  const known = GROUPS.flatMap((g) => g.roles);
  const groups = grouped
    ? [
        ...GROUPS.map((g) => ({ ...g, drugs: drugs.filter((d) => g.roles.includes(d.role)) })),
        { heading: "Other roles", roles: [], drugs: drugs.filter((d) => !known.includes(d.role)) },
      ].filter((g) => g.drugs.length > 0)
    : [{ heading: "", roles: [], drugs }];
  return (
    <section>
      {heading ? <h3 className="mb-1 text-sm font-bold">Drugs</h3> : null}
      <p className="mb-2 text-xs text-gray-500">
        From the GoodOption catalog (LLM-assisted, NCIt classes); verify against the protocol.
      </p>
      <div className="flex flex-col gap-3">
        {groups.map((g) => (
          <div key={g.heading || "all"} className="flex flex-col gap-1">
            {g.heading ? (
              <h4 className="text-xs font-semibold tracking-wider text-gray-500 uppercase">
                {g.heading}
              </h4>
            ) : null}
            <ul className="flex flex-col divide-y divide-gray-100 rounded-md border border-gray-200">
              {g.drugs.map((d) => (
                <DrugItem key={d.drug_id} drug={d} onEvidence={onEvidence} />
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

function DrugItem({ drug: d, onEvidence }: { drug: Drug; onEvidence: OpenEvidence | undefined }) {
  return (
    <li className="flex flex-col gap-0.5 px-3 py-2 text-sm">
      <div className="flex flex-wrap items-baseline gap-x-2">
        <CatalogLink
          kind="drug"
          id={d.drug_id}
          onOpen={onEvidence}
          variant="link"
          className="font-medium"
        >
          {d.name}
        </CatalogLink>
        {d.listed_name ? (
          <span className="text-xs text-gray-500">listed as {d.listed_name}</span>
        ) : null}
        <Badge
          variant={d.role === "investigational" ? "secondary" : "outline"}
          className="ml-auto self-center"
        >
          {ROLE_LABELS[d.role] ?? d.role}
        </Badge>
      </div>
      {d.mechanism_types.length ? (
        <p className="text-gray-700">
          <strong>Type:</strong>{" "}
          <ClassLinks names={d.mechanism_types} ids={d.class_ids} onOpen={onEvidence} />
        </p>
      ) : null}
      {d.payload ? (
        <p className="text-gray-700">
          <strong>Payload:</strong> <PayloadLinks drug={d} onOpen={onEvidence} />
        </p>
      ) : null}
      {d.mechanism ? <p className="text-gray-700">{d.mechanism}</p> : null}
      {d.arms.length ? (
        <p className="text-xs text-gray-500">
          {d.arms.length === 1 ? "Arm" : "Arms"}: {d.arms.join("; ")}
        </p>
      ) : null}
    </li>
  );
}
