export function percent(p: number): string {
  return `${String(Math.round(p * 100))}%`;
}

export function when(seconds: number | null | undefined): string {
  if (!seconds) return "";
  return new Date(seconds * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function isoWhen(iso: string | null | undefined): string {
  if (!iso) return "";
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/** "Lung: non-small cell" and the like for cancer-type keys. */
export function typeLabels(keys: string[], labels: Record<string, string>): string[] {
  const specific = keys.filter(
    (k) => !(k === "lung_any" && keys.some((o) => o.startsWith("lung_") && o !== k)),
  );
  return specific.map((k) => labels[k] ?? k);
}

/** A patient's one-line description: "62 F · Lung adenocarcinoma". */
export function patientLine(p: {
  age?: string;
  sex?: string;
  cancer_type?: string;
  histology?: string;
}): string {
  const who = [p.age?.replace(/\s*years?( old)?/i, ""), p.sex?.charAt(0).toUpperCase()]
    .filter(Boolean)
    .join(" ");
  const what = [p.cancer_type, p.histology].filter(Boolean).join(" · ");
  return [who, what].filter(Boolean).join(" · ");
}
