import type { Metadata } from "next";
import { Heebo } from "next/font/google";
import { Nav } from "@/components/nav";
import "./globals.css";

const heebo = Heebo({
  variable: "--font-heebo",
  subsets: ["hebrew", "latin"],
});

export const metadata: Metadata = {
  title: "Hebrew Toxicity Admin",
  description: "ניהול דאטה, אימון ופרסום של מסווג התגובות הרעילות",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="he" dir="rtl" className={`${heebo.variable} h-full antialiased`}>
      <body className="min-h-full font-sans">
        <div className="mx-auto flex min-h-screen max-w-7xl flex-col md:flex-row">
          <aside className="border-b border-line p-4 md:w-56 md:shrink-0 md:border-b-0 md:border-l md:p-6">
            <div className="mb-4 px-3 md:mb-8">
              <div className="text-base font-semibold">מסווג רעילות</div>
              <div className="text-xs text-muted">ממשק ניהול</div>
            </div>
            <Nav />
          </aside>
          <main className="min-w-0 flex-1 p-4 md:p-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
