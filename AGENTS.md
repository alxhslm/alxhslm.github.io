# Agent Instructions & Project Guidelines

## Overview

This repository contains the source code for the personal website built with **Hugo**.

## Common Commands

- **Start Hugo Development Server**: `hugo server -D` (available on port 1313)
- **Build Production Site**: `hugo --gc --minify`
- **Lint / Pre-commit Hooks**: `pre-commit run --all-files`

## Creating a New Blog Post

Posts use Hugo Page Bundles inside `content/posts/<post-slug>/index.md`:

1. **Generate New Post**:
   ```bash
   hugo new posts/<post-slug>/index.md
   ```
2. **Frontmatter Format** (TOML using `+++`):
   ```toml
   +++
   title = 'My New Post Title'
   date = 2026-07-20T20:00:00Z
   tags = ['tooling', 'devops']
   draft = false
   +++
   ```
3. **Assets**: Store images and media specific to the post inside `content/posts/<post-slug>/`.

## Codebase Organization

- `content/posts/`: Blog posts written in Markdown with Hugo frontmatter.
- `layouts/`: Custom HTML templates.
- `assets/`: Global CSS/JS assets.
- `static/`: Static images, files, and favicon assets.
- `.devcontainer/`: Devcontainer configuration supporting both Claude Code and Antigravity (`agy`).
