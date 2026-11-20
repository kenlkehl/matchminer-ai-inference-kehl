import { type ComponentProps, useEffect, useEffectEvent, useState } from "react";

import { Input } from "@/components/ui/input";

const SEARCH_DEBOUNCE_MS = 300;

/**
 * A search box for a search param: writes the URL once typing pauses. Follows the URL when something else changes it (Back),
 * but not when the URL catches up with what it sent, so typing in the meantime isn't lost.
 * To clear it to a value it may already have (the overview's Clear All), remount it.
 */
export function SearchInput({
  value,
  onChange,
  ...props
}: Omit<ComponentProps<typeof Input>, "value" | "onChange" | "type"> & {
  value: string;
  onChange: (value: string) => void;
}) {
  const [text, setText] = useState(value);
  // The last value written to the URL, and the last value read from it.
  const [sent, setSent] = useState(value);
  const [seen, setSeen] = useState(value);
  if (value !== seen) {
    setSeen(value);
    if (value !== sent) {
      setSent(value);
      setText(value);
    }
  }

  const send = useEffectEvent((next: string) => {
    setSent(next);
    onChange(next);
  });
  useEffect(() => {
    if (text === sent) return;
    const timer = setTimeout(() => {
      send(text);
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
    };
  }, [text, sent]);

  return (
    <Input
      {...props}
      type="search"
      value={text}
      onChange={(e) => {
        setText(e.target.value);
      }}
    />
  );
}
