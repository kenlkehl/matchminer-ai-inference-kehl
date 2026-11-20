import type { ReactNode } from "react";

import { Separator } from "@/components/ui/separator";

/** A page's title and grey description text, ruled off from the page. */
export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header>
      <h1 className="mb-2 text-2xl font-bold">{title}</h1>
      {children ? <div className="space-y-1 text-sm text-gray-600">{children}</div> : null}
      <Separator className="mt-4" />
    </header>
  );
}
