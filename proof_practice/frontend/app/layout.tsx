import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Proof Practice · Jev",
  description: "Practice elementary mathematical proofs with five-part rubric feedback.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
