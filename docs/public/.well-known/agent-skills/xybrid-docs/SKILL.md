---
name: xybrid-docs
description: Find and read authoritative Xybrid SDK documentation for the current task.
---

# Read the Xybrid docs

1. Identify the user's SDK, target platform, and task. Check the version in their project before assuming a documented feature is available.
2. Search the public Xybrid docs MCP server at `https://docs.xybrid.dev/api/mcp` over Streamable HTTP, or browse the [English Markdown index](https://docs.xybrid.dev/llms.txt) or [Chinese Markdown index](https://docs.xybrid.dev/zh/llms.txt).
3. Read the matching quickstart or SDK page before suggesting code. Prefer individual Markdown pages over loading the full documentation dump. Link the source pages in your answer.
4. Distinguish local model execution from opt-in cloud fallback and trace export. Do not suggest authentication or telemetry setup for an offline-only task.
