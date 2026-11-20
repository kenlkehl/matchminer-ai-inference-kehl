"use client";

import * as React from "react";
import { cn } from "cn";

/**
 * The Reflex app's two table looks, besides shadcn's default:
 * - `list`: admin lists; small uppercase grey headers, zebra rows without row borders.
 * - `grid`: the trials overview; every cell bordered, roomy top-aligned cells that wrap, breaking
 *   words longer than the column (inline blocks included) rather than spilling into the next.
 */
type TableVariant = "default" | "list" | "grid";

const TableVariantContext = React.createContext<TableVariant>("default");

const VARIANTS = {
  default: {
    header: "[&_tr]:border-b",
    row: "border-b transition-colors hover:bg-muted/50 has-aria-expanded:bg-muted/50 data-[state=selected]:bg-muted",
    head: "h-10 px-2 align-middle font-medium whitespace-nowrap text-foreground",
    cell: "p-2 align-middle whitespace-nowrap",
  },
  list: {
    header: "bg-gray-50 [&_tr]:border-b",
    row: "even:bg-gray-50",
    head: "px-3 py-2 align-middle text-xs font-semibold whitespace-nowrap text-gray-500 uppercase",
    cell: "px-3 py-2 align-middle",
  },
  grid: {
    header: "bg-brand-blue/5",
    row: "border-b border-gray-300",
    head: "border-r border-gray-300 p-3 align-top font-bold wrap-anywhere text-muted-foreground last:border-r-0",
    cell: "border-r border-gray-300 p-3 align-top wrap-anywhere text-muted-foreground last:border-r-0",
  },
} satisfies Record<TableVariant, Record<"header" | "row" | "head" | "cell", string>>;

function Table({
  className,
  variant = "default",
  ...props
}: React.ComponentProps<"table"> & { variant?: TableVariant }) {
  return (
    <TableVariantContext value={variant}>
      <div data-slot="table-container" className="relative w-full overflow-x-auto">
        <table
          data-slot="table"
          data-variant={variant}
          className={cn(
            "w-full caption-bottom text-sm",
            variant === "grid" && "table-fixed border-t border-gray-300",
            className,
          )}
          {...props}
        />
      </div>
    </TableVariantContext>
  );
}

function TableHeader({ className, ...props }: React.ComponentProps<"thead">) {
  const variant = React.use(TableVariantContext);
  return (
    <thead
      data-slot="table-header"
      className={cn(VARIANTS[variant].header, className)}
      {...props}
    />
  );
}

function TableBody({ className, ...props }: React.ComponentProps<"tbody">) {
  return (
    <tbody
      data-slot="table-body"
      className={cn("[&_tr:last-child]:border-0", className)}
      {...props}
    />
  );
}

function TableFooter({ className, ...props }: React.ComponentProps<"tfoot">) {
  return (
    <tfoot
      data-slot="table-footer"
      className={cn("border-t bg-muted/50 font-medium [&>tr]:last:border-b-0", className)}
      {...props}
    />
  );
}

function TableRow({ className, ...props }: React.ComponentProps<"tr">) {
  const variant = React.use(TableVariantContext);
  return <tr data-slot="table-row" className={cn(VARIANTS[variant].row, className)} {...props} />;
}

function TableHead({ className, ...props }: React.ComponentProps<"th">) {
  const variant = React.use(TableVariantContext);
  return (
    <th
      data-slot="table-head"
      className={cn("text-left [&:has([role=checkbox])]:pr-0", VARIANTS[variant].head, className)}
      {...props}
    />
  );
}

function TableCell({ className, ...props }: React.ComponentProps<"td">) {
  const variant = React.use(TableVariantContext);
  return (
    <td
      data-slot="table-cell"
      className={cn("[&:has([role=checkbox])]:pr-0", VARIANTS[variant].cell, className)}
      {...props}
    />
  );
}

function TableCaption({ className, ...props }: React.ComponentProps<"caption">) {
  return (
    <caption
      data-slot="table-caption"
      className={cn("mt-4 text-sm text-muted-foreground", className)}
      {...props}
    />
  );
}

export { Table, TableHeader, TableBody, TableFooter, TableHead, TableRow, TableCell, TableCaption };
