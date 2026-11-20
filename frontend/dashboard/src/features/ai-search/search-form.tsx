import { useMutation, useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { type ReactNode, useState } from "react";

import { SearchInput } from "@/components/search-input";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { api, type PatientDetail, type TrialsForPatientResult } from "@/lib/api";
import { patientLine } from "@/lib/format";
import { metaOptions, patientsOptions } from "@/lib/queries";

import { CANCER_SUMMARY_SCAFFOLD, setAiState, useAiState } from "./store";

/**
 * mm-dash's AI search form: a patient by ID (here a synthetic patient, chosen from the corpus), or a
 * patient summary with its common exclusions typed in.
 */
export function SearchForm() {
  const ai = useAiState();
  const [tab, setTab] = useState<"patient" | "summary">(
    ai.patientId || ai.summary === CANCER_SUMMARY_SCAFFOLD ? "patient" : "summary",
  );
  const search = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api.post<TrialsForPatientResult>("/api/match/trials-for-patient", body),
    onSuccess: (result) => {
      setAiState((s) => ({
        result,
        page: "results",
        searches: s.searches + 1,
        filters: { ...s.filters, excludeHigherThreshold: true },
      }));
    },
  });
  const submit = (byPatient: boolean) => {
    const common = { k: 50, per_trial: !ai.allCohorts, scope: "dashboard" };
    search.mutate(
      byPatient && ai.patientId
        ? { ...common, patient_id: ai.patientId }
        : { ...common, summary: ai.summary, boilerplate: ai.boilerplate },
    );
  };
  const summaryEmpty = !ai.summary.trim() || ai.summary === CANCER_SUMMARY_SCAFFOLD;

  return (
    <section aria-label="Search" className="rounded-lg border border-gray-200 bg-white">
      <fieldset disabled={search.isPending} className="contents">
        <Tabs
          value={tab}
          onValueChange={(value: "patient" | "summary") => {
            setTab(value);
          }}
        >
          <TabsList variant="line" className="w-full justify-start border-b px-4">
            <TabsTrigger value="patient" className="flex-none">
              Synthetic Patient
            </TabsTrigger>
            <TabsTrigger value="summary" className="flex-none">
              Patient Summary
            </TabsTrigger>
          </TabsList>
          <div className="p-4">
            <TabsContent value="patient">
              <PatientPicker
                onPick={(p) => {
                  setAiState({ patientId: p.id, summary: p.key, boilerplate: p.boilerplate });
                }}
              />
              <Actions
                pending={search.isPending}
                error={search.error?.message}
                label="Search"
                disabled={!ai.patientId}
                onSubmit={() => {
                  submit(true);
                }}
                note={
                  ai.patientId ? (
                    <span className="text-sm text-gray-700">
                      Selected: <strong>{ai.patientId}</strong>
                    </span>
                  ) : (
                    <span className="text-sm text-gray-500">Choose a patient above.</span>
                  )
                }
              />
            </TabsContent>
            <TabsContent value="summary" keepMounted>
              <form
                noValidate
                onSubmit={(e) => {
                  e.preventDefault();
                  submit(false);
                }}
              >
                <div className="grid items-start gap-4 md:grid-cols-[3fr_1fr]">
                  <label className="flex flex-col gap-1">
                    <span className="text-sm font-medium">Cancer Summary</span>
                    <Textarea
                      value={ai.summary}
                      onChange={(e) => {
                        setAiState({ summary: e.target.value, patientId: null });
                      }}
                      className="field-sizing-fixed scrollbar-visible h-[calc(8lh+10px)] font-mono text-sm"
                    />
                    <span className="text-xs text-gray-500">
                      * A high-quality cancer summary includes age, sex, cancer type/primary site,
                      disease histology, current extent of disease, relevant biomarkers, and
                      treatment history.
                    </span>
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-sm font-medium">Common Exclusions</span>
                    <Textarea
                      value={ai.boilerplate}
                      placeholder="Conditions that commonly preclude enrollment (e.g. poor performance status, pneumonitis, active brain metastases)."
                      onChange={(e) => {
                        setAiState({ boilerplate: e.target.value, patientId: null });
                      }}
                      className="field-sizing-fixed scrollbar-visible h-[calc(8lh+10px)] font-mono text-sm placeholder:font-sans placeholder:italic"
                    />
                  </label>
                </div>
                <Actions
                  pending={search.isPending}
                  error={search.error?.message}
                  label="Submit"
                  disabled={summaryEmpty}
                  submit
                  onClear={() => {
                    setAiState({
                      summary: CANCER_SUMMARY_SCAFFOLD,
                      boilerplate: "",
                      patientId: null,
                      result: null,
                    });
                  }}
                  note={
                    ai.patientId ? (
                      <span className="text-xs text-gray-500">
                        From synthetic patient {ai.patientId}; editing it searches the edited text.
                      </span>
                    ) : null
                  }
                />
              </form>
            </TabsContent>
          </div>
        </Tabs>
      </fieldset>
    </section>
  );
}

function Actions({
  pending,
  error,
  label,
  disabled,
  submit = false,
  onSubmit,
  onClear,
  note,
}: {
  pending: boolean;
  error?: string;
  label: string;
  disabled: boolean;
  submit?: boolean;
  onSubmit?: () => void;
  onClear?: () => void;
  note?: ReactNode;
}) {
  return (
    <div className="mt-3 flex flex-col gap-2">
      {error ? <p className="text-sm text-red-700">{error}</p> : null}
      <div className="flex flex-wrap items-center gap-2">
        {note}
        <div className="ml-auto flex gap-2">
          {onClear ? (
            <Button type="button" variant="outline" onClick={onClear}>
              Clear
            </Button>
          ) : null}
          <Button
            type={submit ? "submit" : "button"}
            disabled={disabled}
            onClick={submit ? undefined : onSubmit}
          >
            {pending ? <Spinner /> : null}
            {label}
          </Button>
        </div>
      </div>
    </div>
  );
}

/** Search the synthetic patients by their summaries' text, filter by cancer type, and pick one. */
function PatientPicker({ onPick }: { onPick: (p: PatientDetail) => void }) {
  const ai = useAiState();
  const meta = useQuery(metaOptions()).data;
  const [q, setQ] = useState("");
  const [cancerType, setCancerType] = useState("");
  const [offset, setOffset] = useState(0);
  const list = useQuery(patientsOptions(q, cancerType, "", offset));
  const pick = useMutation({
    mutationFn: (id: string) => api.get<PatientDetail>(`/api/patients/${encodeURIComponent(id)}`),
    onSuccess: onPick,
  });
  const data = list.data;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex min-w-64 flex-1 flex-col gap-1">
          <span className="text-sm font-medium">Search the synthetic patients</span>
          <SearchInput
            value={q}
            placeholder="Words or regular expressions in the summary, e.g. EGFR osimertinib"
            onChange={(value) => {
              setQ(value);
              setOffset(0);
            }}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-medium">Cancer type</span>
          <select
            className="h-9 rounded-md border border-input bg-background px-2 text-sm"
            value={cancerType}
            onChange={(e) => {
              setCancerType(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">Any</option>
            {meta?.cancer_types
              .filter((t) => t.group !== "Basket")
              .map((t) => (
                <option key={t.key} value={t.key}>
                  {t.label}
                </option>
              ))}
          </select>
        </label>
      </div>
      <div className="overflow-hidden rounded-md border border-gray-200">
        {data ? (
          <>
            <ul className="max-h-80 divide-y divide-gray-100 overflow-y-auto">
              {data.rows.map((p) => (
                <li key={p.id}>
                  <button
                    type="button"
                    className={cn(
                      "flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-brand-blue/5",
                      ai.patientId === p.id && "bg-brand-blue/10",
                    )}
                    onClick={() => {
                      pick.mutate(p.id);
                    }}
                  >
                    <span className="flex-1 text-gray-800">{patientLine(p)}</span>
                    <span className="shrink-0 font-mono text-xs text-gray-400">{p.id}</span>
                    {pick.isPending && pick.variables === p.id ? (
                      <Spinner className="size-3.5" />
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
            <div className="flex items-center gap-2 border-t border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600">
              {data.total.toLocaleString()} of {data.of.toLocaleString()} patients
              <div className="ml-auto flex gap-1">
                <Button
                  size="xs"
                  variant="outline"
                  disabled={offset === 0}
                  onClick={() => {
                    setOffset(Math.max(0, offset - 25));
                  }}
                >
                  Previous
                </Button>
                <Button
                  size="xs"
                  variant="outline"
                  disabled={offset + 25 >= data.total}
                  onClick={() => {
                    setOffset(offset + 25);
                  }}
                >
                  Next
                </Button>
              </div>
            </div>
          </>
        ) : list.isError ? (
          <p className="p-3 text-sm text-red-700">{list.error.message}</p>
        ) : (
          <div className="flex justify-center p-6">
            <Spinner />
          </div>
        )}
      </div>
    </div>
  );
}
