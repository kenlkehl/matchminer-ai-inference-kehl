import { Accordion as AccordionPrimitive } from "@base-ui/react/accordion";
import { cn } from "cn";
import { ChevronDownIcon } from "lucide-react";
import type { ReactNode } from "react";

function Accordion({ className, ...props }: AccordionPrimitive.Root.Props) {
  return (
    <AccordionPrimitive.Root
      data-slot="accordion"
      className={cn("flex w-full flex-col", className)}
      {...props}
    />
  );
}

function AccordionItem({ className, ...props }: AccordionPrimitive.Item.Props) {
  return (
    <AccordionPrimitive.Item
      data-slot="accordion-item"
      className={cn("not-last:border-b", className)}
      {...props}
    />
  );
}

/**
 * The item's header row: the trigger's label, then `actions` (links or buttons that don't
 * toggle the item), then the chevron at the far right. The whole row toggles the item.
 */
function AccordionTrigger({
  className,
  children,
  actions,
  ...props
}: AccordionPrimitive.Trigger.Props & { actions?: ReactNode }) {
  return (
    <AccordionPrimitive.Header
      className={cn(
        "group/accordion-header relative flex items-center gap-2 py-4 text-left text-sm font-medium",
        className,
      )}
    >
      <AccordionPrimitive.Trigger
        data-slot="accordion-trigger"
        className="text-left outline-none after:absolute after:inset-0 after:rounded-[inherit] focus-visible:after:ring-3 focus-visible:after:ring-ring/50 aria-disabled:pointer-events-none aria-disabled:opacity-50"
        {...props}
      >
        {children}
      </AccordionPrimitive.Trigger>
      {actions ? <div className="relative flex items-center gap-1">{actions}</div> : null}
      <ChevronDownIcon
        data-slot="accordion-trigger-icon"
        className="pointer-events-none ml-auto size-5 shrink-0 text-gray-700 transition-transform group-data-open/accordion-header:rotate-180"
      />
    </AccordionPrimitive.Header>
  );
}

function AccordionContent({ className, children, ...props }: AccordionPrimitive.Panel.Props) {
  return (
    <AccordionPrimitive.Panel
      data-slot="accordion-content"
      className="overflow-hidden text-sm data-open:animate-accordion-down data-closed:animate-accordion-up"
      {...props}
    >
      <div
        className={cn(
          "h-(--accordion-panel-height) pt-0 pb-4 data-ending-style:h-0 data-starting-style:h-0",
          className,
        )}
      >
        {children}
      </div>
    </AccordionPrimitive.Panel>
  );
}

export { Accordion, AccordionItem, AccordionTrigger, AccordionContent };
