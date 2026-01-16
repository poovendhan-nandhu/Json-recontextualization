⏺ Here's where all the prompt files are located:

  src/
  ├── utils/
  │   └── prompts.py          ← MAIN SHARD PROMPTS (Factsheet, Adaptation, Regeneration)
  │
  ├── stages/
  │   ├── alignment_fixer_prompts.py  ← Alignment fix prompts (KLO-Question, KLO-Resource, etc.)
  │   ├── fixer_prompts.py            ← Semantic fixer prompts (Metrics, Personas, Rubrics)
  │   ├── alignment_checker.py        ← Contains inline prompts for 9 alignment checks
  │   ├── alignment_fixer.py          ← Uses alignment_fixer_prompts.py
  │   └── fixers.py                   ← Uses fixer_prompts.py
  │
  ├── core/                    ← LEAF METHOD (alternative pipeline)
  │   ├── smart_prompts.py     ← Leaf adaptation prompts
  │   ├── leaf_adapter.py      ← Leaf adaptation logic
  │   ├── leaf_fixers.py       ← Leaf fix prompts
  │   ├── leaf_validators.py   ← Leaf validation prompts
  │   └── decider.py           ← Decision prompts
  │
  └── validation/
      └── report_generator.py  ← Report formatting (no LLM prompts)

  Key Prompt Files Summary
  ┌───────────────────────────────────────┬──────────────────────────────────┬───────┐
  │                 File                  │             Purpose              │ Lines │
  ├───────────────────────────────────────┼──────────────────────────────────┼───────┤
  │ src/utils/prompts.py                  │ Main shard adaptation prompts    │ ~766  │
  ├───────────────────────────────────────┼──────────────────────────────────┼───────┤
  │ src/stages/alignment_fixer_prompts.py │ Alignment fix prompts            │ ~200  │
  ├───────────────────────────────────────┼──────────────────────────────────┼───────┤
  │ src/stages/fixer_prompts.py           │ Semantic fixer prompts           │ ~180  │
  ├───────────────────────────────────────┼──────────────────────────────────┼───────┤
  │ src/stages/alignment_checker.py       │ 9 inline alignment check prompts │ ~1400 │
  ├───────────────────────────────────────┼──────────────────────────────────┼───────┤
  │ src/core/smart_prompts.py             │ Leaf method prompts              │ ~300  │
  └───────────────────────────────────────┴──────────────────────────────────┴───────┘
  Two Pipelines

  1. SHARD Pipeline (what you're using):
    - src/utils/prompts.py → main prompts
    - src/stages/*.py → stage-specific prompts
  2. LEAF Pipeline (alternative):
    - src/core/smart_prompts.py → leaf prompts
    - src/core/leaf_*.py → leaf-specific logic

❯