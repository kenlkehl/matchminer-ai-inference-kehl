import { Link, useRouterState } from "@tanstack/react-router";

export type AppTab = "dashboard" | "ai";

/** Which of the applications the current page belongs to. */
export function useAppTab(): AppTab {
  return useRouterState({
    select: (s) => (s.location.pathname.startsWith("/ai") ? "ai" : "dashboard"),
  });
}

const TABS: { tab: AppTab; to: "/" | "/ai"; label: string }[] = [
  { tab: "dashboard", to: "/", label: "Trial Management and Study Status" },
  { tab: "ai", to: "/ai", label: "Search for trials for a patient" },
];

/** The tabs above every page, as in mm-dash. */
export function AppTabs() {
  const current = useAppTab();
  return (
    <nav
      aria-label="Applications"
      className="-mx-6 -mt-6 mb-6 flex border-b border-gray-200 bg-gray-100 px-6 pt-2 print:hidden"
    >
      {TABS.map(({ tab, to, label }) => (
        <Link
          key={tab}
          to={to}
          aria-current={tab === current ? "page" : undefined}
          className="-mb-px flex items-center gap-1.5 rounded-t-md border border-transparent px-4 py-2 text-sm font-semibold text-gray-500 hover:text-gray-900 aria-[current=page]:border-t-2 aria-[current=page]:border-gray-200 aria-[current=page]:border-t-brand-blue aria-[current=page]:border-b-white aria-[current=page]:bg-white aria-[current=page]:text-gray-900"
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
