# Agent Instructions & Project Guidelines

## Overview
This repository contains the source code for the personal website built with **Hugo**.

## Common Commands
- **Start Hugo Development Server**: `hugo server -D` (available on port 1313)
- **Build Production Site**: `hugo --gc --minify`
- **Lint / Pre-commit Hooks**: `pre-commit run --all-files`

## Codebase Organization
- `content/posts/`: Blog posts written in Markdown with Hugo frontmatter.
- `layouts/`: Custom HTML templates.
- `assets/`: Global CSS/JS assets.
- `static/`: Static images, files, and favicon assets.
- `.devcontainer/`: Devcontainer configuration supporting both Claude Code and Antigravity (`agy`).
