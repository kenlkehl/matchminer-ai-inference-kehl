import { useQuery } from "@tanstack/react-query";
import { Link, useRouterState } from "@tanstack/react-router";
import { ActivityIcon, FunnelIcon, type LucideIcon, SearchIcon, TableIcon } from "lucide-react";
import { type ReactNode, useCallback, useState, useSyncExternalStore } from "react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
  sidebarCookieOpen,
  useSidebar,
} from "@/components/ui/sidebar";
import type { CenterInfo, SiteBranding } from "@/lib/api";
import { metaOptions } from "@/lib/queries";
import { useSite } from "@/lib/site";

import { AppTabs, useAppTab } from "./app-tabs";
import { SidebarSlotContext } from "./sidebar-slot";

/** Centers by institution, in the site's order (its site config's institution prefixes); the rest last. */
function byInstitution(centers: CenterInfo[], institutions: SiteBranding["institutions"]) {
  const test = (i: { prefixes: string[] }, name: string) =>
    i.prefixes.some((p) => name.startsWith(p));
  const groups = institutions.map((i) => ({
    label: i.label,
    centers: centers
      .filter((c) => test(i, c.name))
      .sort((a, b) => a.short_name.localeCompare(b.short_name)),
  }));
  const rest = centers.filter((c) => !institutions.some((i) => test(i, c.name)));
  if (rest.length) groups.push({ label: institutions.length ? "Other" : "Centers", centers: rest });
  return groups.filter((g) => g.centers.length > 0);
}

/**
 * The layout: mm-dash's sidebar (navigation, the disease centers, the page's own panel),
 * which collapses to its icons, the application tabs, the page, and the footer.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  const [sidebarOpen] = useState(sidebarCookieOpen);
  const tab = useAppTab();
  const meta = useQuery(metaOptions()).data;
  const site = useSite();

  return (
    <SidebarSlotContext value={slot}>
      <SidebarProvider defaultOpen={sidebarOpen}>
        <Sidebar collapsible="icon">
          <SidebarHeader className="px-4 pt-4 group-data-[collapsible=icon]:items-center group-data-[collapsible=icon]:px-2">
            <div className="flex items-start justify-between gap-2 group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:items-center">
              <div className="group-data-[collapsible=icon]:hidden">
                <p className="text-xl font-bold">MatchMiner</p>
                {site.institution ? (
                  <p className="text-xs text-sidebar-foreground/50">{site.institution}</p>
                ) : null}
              </div>
              <p
                aria-hidden
                className="hidden text-sm font-bold group-data-[collapsible=icon]:block"
              >
                MM
              </p>
              <SidebarTrigger className="text-sidebar-foreground/60 hover:bg-white/10 hover:text-white" />
            </div>
          </SidebarHeader>
          <SidebarContent className="scrollbar-visible">
            <SidebarGroup>
              <SidebarGroupContent>
                <SidebarMenu>
                  {tab === "dashboard" ? (
                    <NavItem to="/" icon={TableIcon} label="Trials Overview" active={false} />
                  ) : (
                    <NavItem to="/ai" icon={SearchIcon} label="Trials for a Patient" active />
                  )}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
            {/* A page's own panel (e.g. its filters) shows only while the sidebar is open. */}
            <div ref={setSlot} className="px-4 group-data-[collapsible=icon]:hidden" />
            <SlotOpener slot={slot} />
            {tab === "dashboard" && meta ? <CenterNav centers={meta.centers} /> : null}
          </SidebarContent>
          <SidebarFooter>
            <SidebarMenu>
              <NavItem to="/status" icon={ActivityIcon} label="Data and Models" active={false} />
            </SidebarMenu>
          </SidebarFooter>
          <SidebarRail />
        </Sidebar>
        <SidebarInset className="min-w-0 p-6">
          <SidebarTrigger className="-mt-3 mb-2 md:hidden print:hidden" />
          <AppTabs />
          {children}
          <AppFooter />
        </SidebarInset>
      </SidebarProvider>
    </SidebarSlotContext>
  );
}

/** Each disease center's dashboard, by institution. */
function CenterNav({ centers }: { centers: CenterInfo[] }) {
  const site = useSite();
  const current = useRouterState({
    select: (s) =>
      s.location.pathname === "/"
        ? ((s.location.search as { center?: string }).center ?? null)
        : null,
  });
  return (
    <>
      {byInstitution(centers, site.institutions).map((group) => (
        <SidebarGroup key={group.label} className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupLabel className="tracking-wider text-sidebar-foreground/50 uppercase">
            {group.label}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {group.centers.map((center) => (
                <SidebarMenuItem key={center.id}>
                  <SidebarMenuButton
                    size="sm"
                    isActive={current === center.id}
                    title={center.name}
                    className="pr-8 text-sidebar-foreground/80 data-active:bg-white/20 data-active:text-white"
                    render={
                      <Link to="/" search={{ center: center.id }}>
                        <span className="truncate">{center.short_name}</span>
                      </Link>
                    }
                  />
                  <SidebarMenuBadge className="text-sidebar-foreground/50">
                    {center.trials}
                  </SidebarMenuBadge>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      ))}
    </>
  );
}

/** While the sidebar is collapsed, a button that opens it on the page's panel, if it has one. */
function SlotOpener({ slot }: { slot: HTMLElement | null }) {
  const { setOpen } = useSidebar();
  const subscribe = useCallback(
    (onChange: () => void) => {
      if (!slot) return () => undefined;
      const observer = new MutationObserver(onChange);
      observer.observe(slot, { childList: true, subtree: true });
      return () => {
        observer.disconnect();
      };
    },
    [slot],
  );
  const hasPanel = useSyncExternalStore(subscribe, () => (slot?.children.length ?? 0) > 0);
  if (!hasPanel) return null;
  return (
    <SidebarGroup className="hidden group-data-[collapsible=icon]:flex">
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton
            tooltip="Filters"
            className="text-sidebar-foreground/80"
            onClick={() => {
              setOpen(true);
            }}
          >
            <FunnelIcon />
            <span>Filters</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarGroup>
  );
}

function AppFooter() {
  const site = useSite();
  return (
    <div className="mt-auto pt-10">
      <footer className="flex flex-wrap justify-between gap-x-6 gap-y-1 border-t border-gray-200 pt-3 text-xs text-gray-500">
        <p className="italic">
          {site.confidentiality ? `${site.confidentiality} ` : ""}Research use only; patients shown
          are synthetic.
        </p>
        {site.institution ? (
          <p>
            {site.institution} – {new Date().getFullYear()}
          </p>
        ) : null}
      </footer>
    </div>
  );
}

interface NavItemProps {
  to: "/" | "/ai" | "/status";
  icon: LucideIcon;
  label: string;
  active: boolean;
}

function NavItem({ to, icon: Icon, label, active }: NavItemProps) {
  const isActive = useRouterState({
    select: (s) =>
      active ||
      (s.location.pathname === to &&
        (to !== "/" || !(s.location.search as { center?: string }).center)),
  });
  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        isActive={isActive}
        tooltip={label}
        className="text-sidebar-foreground/80 data-active:bg-white/20 data-active:text-white"
        render={
          <Link to={to}>
            <Icon />
            <span>{label}</span>
          </Link>
        }
      />
    </SidebarMenuItem>
  );
}
