import { createFileRoute } from "@tanstack/react-router";

import { StatusPage } from "@/features/status/status-page";

export const Route = createFileRoute("/_app/status")({
  component: StatusPage,
});
