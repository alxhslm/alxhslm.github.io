+++
title = 'My LaTeX devcontainer broke. So I tried Typst instead.'
date = 2026-04-08T00:00:00Z
tags = ['development']
+++

I used LaTeX at university and loved how it let me focus on the *meaning* of my writing rather than fiddling with formatting. That principle still holds. But everything else about LaTeX has aged poorly.

When I went to update my CV recently (just keeping it current, not job hunting!), my LaTeX devcontainer had broken. TeX Live had incremented versions and nothing would compile. Rather than debug it, I decided to try [Typst](https://typst.app/); a modern alternative I'd seen on my GitHub feed months ago.

The migration took 30 minutes thanks to Claude Code. Within an hour, I was wondering why I'd spent so much time working around LaTeX's limitations.

## What's wrong with LaTeX?

### Installation is a nightmare

TeX Live is **massive**; multiple gigabytes for the full distribution. Building a devcontainer with LaTeX takes painfully long because you're forced to install the entire ecosystem even if you only need 5% of it. There's no lightweight option for simple documents.

For my CV, this was absurd. I don't need the thousand obscure packages that come with a full TeX installation.

### The tooling ecosystem is stuck in the 90s

The tooling around LaTeX feels ancient because, well, it is:

- **latexindent is a random Perl script**; not a properly maintained tool with a clear roadmap
- No modern language servers that work reliably
- Linters are an afterthought
- IDE support is hacky at best

Compare this to any modern programming language where you get proper LSP support, formatting, and linting out of the box.

### No proper package management

This one really frustrated me. There's **no way to pin package versions** in LaTeX. None. Your document might break when packages update, and you have no way to prevent it.

I hacked together my own solution using a `requirements.txt` file with a list of package names, but even that doesn't support version pinning. For a tool that's supposed to enable reproducible documents, this is frankly ridiculous.

### Multiple recompiles for bibliographies

Want citations? Hope you enjoy compiling your document **three times**:

1. First pass to build the document structure
2. Second pass to process references
3. Third pass to resolve citations

It's 2025. Why is this still a thing?

## What Typst gets right

### Lightning fast compilation

Typst compiles **instantly**. The live preview actually works. When I'm iterating on formatting, the tight feedback loop makes a huge difference.

LaTeX compilation times might not seem like a big deal until you've experienced the alternative.

### Modern tooling that actually works

- Proper language server support
- Real linters that catch errors before compilation
- Works with modern editors out of the box
- Error messages that are actually helpful

It feels like a tool built in the 2020s, not the 1980s.

### One compile pass

Bibliographies? Citations? Cross-references? Everything updates in a **single compilation**. No multiple passes, no running special commands. It just works.

### Sensible package management

Typst has a clear dependency model with actual versioning. Reproducible builds work without custom hacks.

## When should you still use LaTeX?

To be fair, LaTeX still wins in some scenarios:

- **Obscure academic requirements**; if you need very specific formatting packages that only exist in LaTeX
- **Collaboration with LaTeX-only colleagues**; sometimes you don't get to choose the tooling
- **Journal templates**; some journals only accept LaTeX submissions
- **Complex typesetting** with very particular requirements such as graphics

For my CV? None of these applied. I just needed clean formatting and reproducible builds.

## How I switched

### The migration was trivial

Historically, migrating from LaTeX would have been painful. Manual rewriting, learning new syntax, recreating all your formatting.

**But thanks to Claude Code, the switching cost is now tiny.**

I just:

1. Gave Claude my LaTeX CV
2. Asked it to convert to Typst
3. Iterated on the output

The whole thing took maybe 30 minutes. Claude handled the syntax conversion and even suggested better ways to structure things in Typst.

This changes the economics completely. The barrier to trying a better tool is no longer "weeks of migration work"; it's "give Claude your files and see if the new tool is better."

### What I gained

Switching to Typst gave me:

- **10x faster compilation**; instant feedback instead of waiting
- **Actual linters** that catch errors before I compile
- **Devcontainer builds in seconds** instead of minutes
- **One compile pass** instead of three for bibliographies
- **Reproducible builds** without custom hacks

What surprised me most was how little I missed from LaTeX. The error messages are clearer, the feedback loop is faster, and I'm not constantly working around tooling issues.

## The broader lesson

LaTeX became the default because that's what people like me learned to use. Academics learned it in grad school, so they teach it. Journals accept it, so people use it. However, that means there is a network effect, and doesn't mean it's chosen on merit.

**Incumbents should always be challenged.** Just because something is established doesn't mean it's the best tool for the job today.

This is more feasible than ever because **AI tools have collapsed the switching cost**. What used to take weeks now takes 30 minutes.

**Staying aware of new tools is part of this.** I saw Typst on my GitHub feed months ago and filed it away. When the right opportunity came, I finally tried it.

## Should you switch?

If you're setting up some new tool, even if it's as simple as a dev container, ask yourself: **am I solving the right problem?** Or am I working around the incumbent's limitations because I haven't challenged whether it's still the best choice?

For me, the answer was clear. I'd been optimising the wrong thing with my LaTeX CI setup. For simple documents with modern workflows, Typst is demonstrably better than LaTeX on every dimension that mattered to me. Not just "good enough"; actually better.
