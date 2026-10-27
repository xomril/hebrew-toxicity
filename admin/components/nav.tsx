"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cx } from "./ui";

const links = [
  { href: "/", label: "סקירה" },
  { href: "/dataset", label: "דאטה-סט" },
  { href: "/review", label: "תור תיוג" },
  { href: "/training", label: "אימון" },
  { href: "/models", label: "מודלים ופרסום" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex gap-1 overflow-x-auto md:flex-col">
      {links.map((link) => {
        const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
        return (
          <Link
            key={link.href}
            href={link.href}
            className={cx(
              "whitespace-nowrap rounded-lg px-3 py-2 text-sm transition",
              active ? "bg-accent/10 font-medium text-accent" : "text-muted hover:bg-hover hover:text-fg",
            )}
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
