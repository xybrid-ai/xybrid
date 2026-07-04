# CLAUDE.md — Claude-specific configuration for xybrid

@AGENTS.md

**All project context and decisions live in [`AGENTS.md`](./AGENTS.md)**
(imported above): what xybrid is, workspace layout, error handling, async
runtime, the SDK run-surface rule, testing, concurrency, the release ritual,
things to leave alone, and open questions. That file is agent-agnostic — every
AI agent reads it, and **new project decisions go there, not here**.

This file holds only Claude-specific material.

## Claude-specific notes

- The `@AGENTS.md` line above is a Claude Code import directive; other agents
  read `AGENTS.md` directly. Keep the import as the first line of content so
  the project guide always loads.
- Project skills live in `.claude/skills/` (e.g. `test-model` for end-to-end
  model verification, `xybrid-init` for generating `model_metadata.json`).
  Prefer them over ad-hoc equivalents when they match the task.
