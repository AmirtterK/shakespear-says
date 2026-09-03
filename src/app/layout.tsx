import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Shakespeare Style Generator",
  description: "A local Shakespeare-style text generator trained from the provided dataset.",
  icons: {
    icon: "/shakespeare.jpg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
