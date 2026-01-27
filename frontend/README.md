# Cartedo Pipeline UI

Simple black & white shadcn-style frontend for the JSON recontextualization pipeline.

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Usage

1. Enter the API WebSocket URL (default: `ws://localhost:8000`)
2. Enter your scenario prompt
3. Paste your input JSON
4. Click "Run Pipeline"

The UI will show:
- Real-time pipeline progress
- Validation scores from all 8 agents
- Final pass/fail decision
- Download button for the adapted JSON

## Stack

- Next.js 14
- TypeScript
- Tailwind CSS
- shadcn/ui components (black & white theme)
