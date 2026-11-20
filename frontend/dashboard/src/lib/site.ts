import { useQuery } from "@tanstack/react-query";

import type { SiteBranding } from "@/lib/api";
import { metaOptions } from "@/lib/queries";

/** The site's branding (the server's site config, dashboard.branding); neutral until the meta loads. */
export const NO_SITE: SiteBranding = {
  institution: "",
  confidentiality: "",
  pi_label: "Site PI",
  higher_threshold_label: "Exclude higher-threshold programs",
  institutions: [],
  trial_links: [],
};

export function useSite(): SiteBranding {
  return useQuery(metaOptions()).data?.site ?? NO_SITE;
}
