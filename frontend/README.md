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

## Deploy to Heroku

```bash
# From frontend directory
cd frontend

# Create Heroku app
heroku create your-app-name

# Deploy
git subtree push --prefix frontend heroku main

# Or if deploying from root repo with subtree:
git subtree split --prefix frontend -b frontend-deploy
git push heroku frontend-deploy:main
```

**Alternative: Deploy as separate repo**
```bash
cd frontend
git init
git add .
git commit -m "Initial commit"
heroku create your-app-name
git push heroku main
```

## Stack

- Next.js 14
- TypeScript
- Tailwind CSS
- shadcn/ui components (black & white theme)
