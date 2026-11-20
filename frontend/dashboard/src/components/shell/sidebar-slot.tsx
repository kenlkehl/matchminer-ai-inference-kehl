import { createContext, type ReactNode, use } from "react";
import { createPortal } from "react-dom";

/** The element in the app sidebar where the current page can put its own panel. */
export const SidebarSlotContext = createContext<HTMLElement | null>(null);

/** Render children into the app sidebar, below the navigation (e.g. the overview's filters). */
export function SidebarSlot({ children }: { children: ReactNode }) {
  const slot = use(SidebarSlotContext);
  return slot ? createPortal(children, slot) : null;
}
