import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Vernacular",
  description: "Real-time voice translation with personality preservation",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
