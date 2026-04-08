+++
title = 'My LaTeX devcontainer broke. So I tried Typst instead.'
date = 2026-04-08T00:00:00Z
tags = ['development']
+++

## The trigger: a new TeX Live release

I used LaTeX at university and loved how it let me focus on the _meaning_ of my writing rather than fiddling with formatting. That principle still holds. But everything else about LaTeX has aged poorly.

When I went to update my CV recently (just keeping it current, not job hunting!), my LaTeX devcontainer had broken. I'd previously written about [setting up CI for a LaTeX CV]({{< ref "/posts/latex" >}}), so this was a setup I'd invested real effort into. TeX Live had incremented versions and nothing would compile.

Rather than debug it, I decided to try [Typst](https://typst.app/) — a modern alternative I'd seen on my GitHub feed months ago. I gave Gemini CLI my LaTeX CV, asked it to convert it, and iterated on the output. The whole thing took 30 minutes. Within an hour of starting, I was wondering why I'd spent so much time working around LaTeX's limitations.

## Why Typst is better

### Compilation is instant

Typst compiles fast enough that the live preview actually works. LaTeX compilation times didn't seem like a big deal until I experienced the alternative — the tight feedback loop when iterating on formatting makes a real difference.

### The tooling is modern

LaTeX tooling feels ancient because it is:

- **latexindent is a Perl script** with no clear roadmap
- Language server support is unreliable
- Error messages are often cryptic

Typst has proper LSP support, real linters, and error messages that actually tell you what went wrong. It feels like a tool built in the 2020s.

### Package management actually works

There's **no way to pin package versions** in LaTeX. Your document might break when packages update and you have no way to prevent it. I hacked together a workaround using a `requirements.txt` file, but even that doesn't support version pinning.

Typst has a proper dependency model with versioning. Reproducible builds work without custom hacks.

### One compile pass

Want citations in LaTeX? You're compiling at least twice — once for the document structure, once to resolve references. Typst handles everything in a single pass.

## The broader point

LaTeX became the default because academics learned it in grad school, taught it to the next generation, and journals standardised on it. That's network effects, not merit. For journal submissions or collaboration with LaTeX-only colleagues, you may not have a choice. For everything else, it's worth asking whether it's still the right tool.

What made it easy to ask that question here was that **AI tools have collapsed the switching cost**. What used to mean weeks of manual rewriting now takes 30 minutes. The economics of trying something better have changed completely.

I saw Typst on my GitHub feed months before I tried it. The devcontainer breaking was the nudge I needed. If you're hitting friction with an incumbent tool, the bar for trying an alternative is lower than it used to be — it's probably worth it.
