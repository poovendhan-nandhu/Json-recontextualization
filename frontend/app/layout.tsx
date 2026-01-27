import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Cartedo Pipeline',
  description: 'JSON Recontextualization Pipeline',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white">{children}</body>
    </html>
  )
}
