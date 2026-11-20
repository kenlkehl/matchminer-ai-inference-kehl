import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";

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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { patientLine } from "@/lib/format";
import { notesOptions, patientOptions } from "@/lib/queries";

/** A synthetic patient's summary and raw notes, and optionally a pair view (e.g. the criteria check). */
export function PatientDialog({
  patientId,
  onClose,
  pairTab,
  defaultTab = "summary",
}: {
  patientId: string;
  onClose: () => void;
  pairTab?: { label: string; content: ReactNode };
  defaultTab?: "summary" | "notes" | "pair";
}) {
  const patient = useQuery(patientOptions(patientId));
  const p = patient.data;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="h-[90svh] sm:max-w-4xl" bodyClassName="gap-4 auto-rows-min">
        <DialogHeader>
          <DialogTitle>
            {p?.custom ? (p.label ?? "Entered summary") : `Synthetic patient ${patientId}`}
          </DialogTitle>
          <DialogDescription className="text-gray-600">{p ? patientLine(p) : ""}</DialogDescription>
        </DialogHeader>
        {p ? (
          <Tabs
            defaultValue={pairTab ? defaultTab : defaultTab === "pair" ? "summary" : defaultTab}
          >
            <TabsList>
              {pairTab ? <TabsTrigger value="pair">{pairTab.label}</TabsTrigger> : null}
              <TabsTrigger value="summary">Summary</TabsTrigger>
              {!p.custom ? <TabsTrigger value="notes">Notes</TabsTrigger> : null}
            </TabsList>
            {pairTab ? (
              <TabsContent value="pair" className="pt-3">
                {pairTab.content}
              </TabsContent>
            ) : null}
            <TabsContent value="summary" className="flex flex-col gap-4 pt-3">
              <Section label="Cancer history">
                <p className="font-mono text-sm break-words whitespace-pre-wrap">{p.key}</p>
              </Section>
              <Section label="Other relevant history (common exclusions)">
                {p.boilerplate.trim() ? (
                  <p className="font-mono text-sm break-words whitespace-pre-wrap">
                    {p.boilerplate}
                  </p>
                ) : (
                  <p className="text-sm text-gray-500 italic">None</p>
                )}
              </Section>
            </TabsContent>
            {!p.custom ? (
              <TabsContent value="notes" className="pt-3">
                <Notes patientId={patientId} />
              </TabsContent>
            ) : null}
          </Tabs>
        ) : patient.isError ? (
          <p className="text-sm text-red-700">{patient.error.message}</p>
        ) : (
          <div className="flex justify-center py-12">
            <Spinner className="size-6" />
          </div>
        )}
        <DialogFooter className="pt-1">
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-1">
      <h3 className="text-xs font-medium tracking-wide text-gray-500 uppercase">{label}</h3>
      {children}
    </section>
  );
}

function Notes({ patientId }: { patientId: string }) {
  const notes = useQuery(notesOptions(patientId));
  if (!notes.data) {
    return notes.isError ? (
      <p className="text-sm text-red-700">{notes.error.message}</p>
    ) : (
      <Spinner className="size-5" />
    );
  }
  if (notes.data.state !== "ready") {
    return (
      <p className="flex items-center gap-2 text-sm text-gray-500">
        <Spinner className="size-4" /> {notes.data.message}
      </p>
    );
  }
  return (
    <ol className="flex flex-col gap-3">
      {notes.data.notes.map((note, i) => (
        <li key={i} className="rounded-md border border-gray-200">
          <p className="border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs font-medium text-gray-600">
            {note.date ? String(note.date).slice(0, 10) : "Undated"}
            {note.type ? ` · ${note.type}` : ""}
          </p>
          <p className="px-3 py-2 font-mono text-xs whitespace-pre-wrap text-gray-800">
            {note.text}
          </p>
        </li>
      ))}
    </ol>
  );
}
