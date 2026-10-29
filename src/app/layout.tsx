import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Shakespeare Style Generator",
  description:
    "A local Shakespeare-style text generator trained from the provided dataset.",
  icons: {
    icon: "/shakespeare.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en" className="dark" data-theme="dark">
      <body className="min-h-screen bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
