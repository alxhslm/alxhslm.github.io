+++
title = 'My LaTeX devcontainer broke. So I tried Typst instead.'
date = 2025-03-08T23:55:00Z
tags = ['development']
+++

## The trigger: a new TeX Live release

I used LaTeX at university and loved how it let me focus on the _meaning_ of my writing rather than fiddling with formatting. That principle still holds. But my software practices have grown since then; I've come to rely on things like formatters, linters, and proper package management, and LaTeX's equivalents don't really measure up.

When I went to update my CV recently (just keeping it current, not job hunting!), my LaTeX devcontainer had broken. I'd previously written about [setting up CI for a LaTeX CV]({{< ref "/posts/latex" >}}), so this was a setup I'd invested real effort into. TeX Live cuts a new annual release and drops the old one from mirrors; my devcontainer was still trying to download the 2025 release, which was gone.

Rather than debug it, I decided to try [Typst](https://typst.app/), a modern alternative I'd seen on my GitHub feed months ago. I was trialling Gemini CLI at the time, so I gave it this task: convert my LaTeX CV to Typst. I iterated on the output. The whole thing took 30 minutes. Within an hour of starting, I was wondering why I'd spent so much time working around LaTeX's limitations.

## Why Typst is better

### Compilation is instant

My CV was taking around 10 seconds to compile in LaTeX; not terrible, but enough to break flow. Typst is fast enough that you don't notice it. The live preview actually works, and iterating on formatting feels completely different when the feedback is immediate.

### The tooling is modern

LaTeX does have language servers, formatters, and linters, but they're niche and under-invested. Most LaTeX users are academics who don't care about this stuff, so the ecosystem reflects that. `latexindent` is a Perl script maintained by one person; language server support exists but is nowhere near the quality you'd get with a mainstream programming language.

Typst is built for people who do care. LSP support, linters, and helpful error messages are first-class. It feels like a tool built in the 2020s.

### Package management actually works

There's **no way to pin package versions** in LaTeX. Your document might break when packages update and you have no way to prevent it. I hacked together a workaround using a `requirements.txt` file, but even that doesn't support version pinning.

Typst has a proper dependency model with versioning. Reproducible builds work without custom hacks.

### One compile pass

Want citations in LaTeX? You're compiling at least twice: once for the document structure, once to resolve references. Typst handles everything in a single pass.

## The broader point

LaTeX became the default because academics learned it at university, taught it to the next generation, and journals standardised on it. That's down to network effects, not merit. For journal submissions or collaboration with LaTeX-only colleagues, you may not have a choice. For everything else, it's worth asking whether it's still the right tool.

What made it easy to ask that question here was that **AI tools have collapsed the switching cost**. What used to mean weeks of manual rewriting now takes 30 minutes. The economics of trying something better have changed completely.

I saw Typst on my GitHub feed months before I tried it. The devcontainer breaking was the nudge I needed. If you're hitting friction with an incumbent tool, the bar for trying an alternative is lower than it used to be; it's worth a try.
