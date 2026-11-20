import { createFileRoute } from "@tanstack/react-router";

import { AiSearchPage } from "@/features/ai-search/ai-search-page";
import { validateAiSearch } from "@/features/ai-search/store";

export const Route = createFileRoute("/_app/ai/")({
  validateSearch: validateAiSearch,
  component: AiSearchPage,
});
