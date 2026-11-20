// Format completed, currently displayed reviews without changing their findings.
import {extractionStatusLabels, isExtractionReview} from "./extraction_view.mjs";
export const statusLabels = {
  completed: "Completed", partially_completed: "Partially completed", planned: "Planned / pending",
  not_done: "Explicitly not done", not_documented: "Not documented", unclear: "Unclear / conflicting",
  error: "Review failed",
};
export const applicabilityLabels = {
  applies: "Applies", not_applicable: "Not applicable", uncertain: "Uncertain",
};

export function isAgentic(review) {
  return review?.metadata?.method === "agentic" || review?.metadata?.scope === "searched_excerpts";
}
export function isCompressed(review) {
  return Boolean(review?.metadata?.compression);
}
export function reviewMethodLabel(review) {
  if (isExtractionReview(review)) return review.metadata.method === "extraction_sql"
    ? "SQL agent over extraction records" : "Structured extraction queries";
  const method = !isAgentic(review) ? "Full-note review"
    : review?.metadata?.followup_review?.max_review_passes > 0
      ? "Agentic note search · coverage review" : "Agentic note search · selected excerpts";
  return method + (isCompressed(review) ? " · compressed notes" : "");
}
export function workupStatusLabel(item, review) {
  if (isExtractionReview(review)) return extractionStatusLabels[item.status] || item.status;
  if (isAgentic(review) && item.status === "not_documented") {
    if (isCompressed(review)) {
      return item?.metadata?.followup_review?.full_record?.complete
        ? "Not documented in compressed notes" : "Not located in searched compressed excerpts";
    }
    return item?.metadata?.followup_review?.full_record?.complete
      ? "Not documented in supplied notes" : "Not located in searched excerpts";
  }
  if (isCompressed(review) && item.status === "not_documented") {
    return "Not documented in compressed notes";
  }
  return statusLabels[item.status] || item.status;
}
export function workupReviewDetail(item, review) {
  const audit = item?.metadata?.followup_review;
  if (!audit) return "";
  const parts = [];
  if (audit.full_record?.complete) {
    parts.push("Full-record fallback reviewed all supplied notes.");
  } else if (audit.full_record) {
    parts.push("Full-record fallback stopped before all supplied notes were reviewed.");
  }
  parts.push(audit.status === "incomplete"
    ? "Follow-up review incomplete; see limitations." : "Search coverage review finished.");
  const detail = parts.join(" ");
  return isCompressed(review) ? detail.replaceAll("supplied notes", "compressed notes") : detail;
}
export function workupBackupDetail(metadata) {
  if (metadata?.backup_resolution_failed) return "The configured backup endpoint could not be prepared; see limitations.";
  if (!metadata?.backup_used) return "";
  const events = metadata.backup_events || [];
  if (!events.length) {
    const count = metadata.backup_items || 0;
    return `Backup model used for ${count} workup ${count === 1 ? "item" : "items"}; see each item's details.`;
  }
  return events.map((event) => {
    const reason = event.reason === "consecutive_python_errors" ? "consecutive Python errors" : "failed request or validation attempts";
    return `Backup model ${event.backup_model} used after ${event.failure_threshold} ${reason} with ${event.primary_model}.`;
  }).join(" ");
}
export function automaticExcerpts(item, review) {
  return (item?.metadata?.evidence_selection || review?.metadata?.evidence_selection)
    === "automatic_reviewed_excerpts";
}
export function patientEvidenceLabel(item, review) {
  if (isExtractionReview(review)) return "Extracted-item evidence";
  if (isCompressed(review)) return automaticExcerpts(item, review)
    ? "Reviewed compressed-note excerpts" : "Compressed-note excerpts";
  return automaticExcerpts(item, review) ? "Reviewed patient-note excerpts" : "Patient-note evidence";
}
export function patientEvidenceCountLabel(item, review) {
  const count = item.evidence?.length || 0;
  const noun = automaticExcerpts(item, review) ? "fragment" : "quote";
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}
export function missingEvidenceLabel(item, review) {
  if (isExtractionReview(review)) return "No extracted item quotes for this result.";
  if (item.status === "error") return "No documentation assessment available.";
  if (isCompressed(review)) return automaticExcerpts(item, review)
    ? "No compressed-note excerpts retained for review."
    : "No supporting evidence located in the compressed notes.";
  return automaticExcerpts(item, review) ? "No patient-note excerpts retained for review."
    : "No supporting evidence located in the supplied notes.";
}

function markdownText(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/([\\`*_{}\[\]|#+.!-])/g, "\\$1")
    .replace(/[\r\n]+/g, " ");
}

export function buildExportReport(result, reviews = {}) {
  const sections = [];
  result.spaces.forEach((space, index) => {
    const review = reviews[index];
    if (!review?.assessments?.length) return;
    const metadata = review.metadata || {};
    const lines = [
      `### ${markdownText(space.rank ?? index + 1)}. ${markdownText(space.name)}`,
      "",
      `**Guideline:** ${markdownText(space.source.title)} · ${markdownText(space.source.version)}`,
      "",
      `**Space ID:** ${markdownText(space.space_trial_id)}`,
      "",
    ];
    if (review.notice) lines.push(markdownText(review.notice), "");
    lines.push(`**Review method:** ${markdownText(reviewMethodLabel(review))}`, "");
    if (metadata.note_count != null) {
      const label = isExtractionReview(review) ? "Notes extracted" : isCompressed(review) ? (isAgentic(review)
        ? "Compressed notes available to search" : "Compressed notes reviewed")
        : isAgentic(review) ? "Raw notes available to search" : "Raw notes reviewed";
      lines.push(`**${label}:** ${markdownText(metadata.note_count)}`, "");
    }
    if (metadata.model) lines.push(`**Review model:** ${markdownText(metadata.model)}`, "");
    const backup = workupBackupDetail(metadata);
    if (backup) lines.push(markdownText(backup), "");
    if (metadata.compression) {
      lines.push(`**Compression:** ${markdownText(metadata.compression.input_characters)} → ${markdownText(metadata.compression.output_characters)} characters; losslessness unverified.`, "");
    }
    for (const [itemIndex, item] of review.assessments.entries()) {
      lines.push(
        `#### ${itemIndex + 1}. ${markdownText(item.name)}`,
        "",
        `**Status:** ${markdownText(workupStatusLabel(item, review))}`,
        "",
        `**Applicability:** ${markdownText(applicabilityLabels[item.applicability] || item.applicability)}`,
        "",
        markdownText(item.bottom_line),
        "",
      );
      const detail = workupReviewDetail(item, review);
      const itemBackup = workupBackupDetail(item.metadata);
      if (itemBackup) lines.push(markdownText(itemBackup), "");
      if (detail) lines.push(`**Review coverage:** ${markdownText(detail)}`, "");
      if (item.limitations?.length) {
        lines.push("**Limitations:** " + item.limitations.map(markdownText).join("; "), "");
      }
      if (item.evidence?.length) {
        lines.push(`**${patientEvidenceLabel(item, review)}**`, "");
        if (isCompressed(review)) {
          lines.push("LLM-compressed text; confirm findings against the original notes.", "");
        }
        if (automaticExcerpts(item, review)) {
          lines.push("Automatically retained review context; each excerpt is not necessarily support for the assessment.", "");
        }
        for (const evidence of item.evidence) {
          lines.push(
            `Note ${markdownText(evidence.note_number)} · ${markdownText(evidence.note_date || "Date unavailable")}${evidence.note_type ? " · " + markdownText(evidence.note_type) : ""}`,
            "",
            ...String(evidence.quote).split(/\r?\n/).map(line => `> ${markdownText(line)}  `),
            "",
          );
        }
      } else {
        lines.push(missingEvidenceLabel(item, review), "");
      }
    }
    sections.push(lines.join("\n"));
  });
  if (!sections.length) return result.report;
  return result.report.trimEnd() + "\n\n---\n\n## Guideline workup concordance review\n\n"
    + "Completed documentation reviews available for the current patient and guideline spaces at export time.\n\n"
    + sections.join("\n");
}
