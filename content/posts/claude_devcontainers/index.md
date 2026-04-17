+++
title = 'Claude Code and devcontainers: the missing link'
date = 2026-04-17T21:00:00Z
tags = ['tooling', 'devops', 'agents']
+++

I've [written before]({{< ref "/posts/devcontainers" >}}) about how much I like devcontainers; they give you reproducible isolated environments, complete isolation between projects and onboarding that takes minutes instead of days. I am convinced they are the right way to manage development environments.

However, a new actor has entered the picture, which doesn't fit neatly into the model devcontainers were designed around — Claude Code.

## The two-body problem

Devcontainers were built around one mental model: _everything_ executes inside the container. VS Code attaches to it, your terminal runs inside it, your tools are installed inside it. The container is your whole world.

Claude Code has the opposite assumption. It's an agent running on your host machine, shelling out commands directly. It expects to be able to reach your filesystem, your credentials, your Docker daemon. It was designed to operate _alongside_ your environment, not inside it.

These two mental models conflict. And every approach you try to reconcile them runs into that conflict eventually.

## Approach 1: install Claude inside the container

The obvious starting point. If Claude needs to run commands in the container, just put Claude in the container. This is what I tried first.

In practice this is more painful than it sounds. A few problems compound quickly:

**Session history breaks.** Claude Code keys session history to the host path. A bind mount won't save you if the path inside the container differs from the host (say, `/home/alex/project` vs `/workspace/project`). You lose continuity.

**Credentials don't transfer.** This one is quite annoying. Currently you can't pass Claude credentials from the host to the container like you can with Git credentials. This means you need to reauthenticate every time the container is rebuilt.

Both of those are solved problems on the host. The container throws that context away.

**Image bloat and maintenance.** Claude Code needs to be installed in every image, even though it has nothing to do with your project's actual environment. Every time Claude updates, every Dockerfile needs updating too.

**Docker daemon access needs managing.** If Claude needs to run `docker build` or manage containers, from inside a container that means Docker-in-Docker (fragile) or Docker-from-Docker (better).

**Multi-container projects suffer.** In a typical `docker-compose` setup — app, database, cache — Claude inside one container can only naturally reach that container's environment. Claude on the host can reach all of them.

Running Claude inside the container is tractable, but you're fighting the tool the whole way.

## Approach 2: a command prefix stopgap

The alternative is to keep Claude on the host but make it operate inside the container's environment. This immediately solves the issues with credentials, session history, and Docker access. The question is whether you can route Claude's shell commands through the container without it needing to live there.

It turns out you can. Claude Code supports hooks that let you intercept tool calls — including bash commands. By prefixing all shell execution with `docker exec -it <container>`, Claude runs its commands inside the container while staying on the host.

I built a proof of concept that works by registering a Claude hook that forwards shell commands into the running container.

{{< github repo="alxhslm/claude-devcontainer" >}}

What surprised me was how seamlessly it works in practice — Claude has no idea its commands are being intercepted and redirected. From its perspective, it's just running commands and getting results. In practice, Claude can run your test suite, execute build commands, and interact with the container's installed tooling — all while keeping host-level access to credentials and the Docker daemon. For local development, this mostly solves the problem today.

The one genuinely unsolved piece is the container lifecycle. The devcontainer only spins up when VS Code opens it. I handled this through the `CLAUDE.md` file: Claude is instructed to check whether the container is running at the start of a session and ask the user to start it if not. It works, but it's a paper-thin solution — the kind of thing that belongs in a `CLAUDE.md` workaround rather than something you should ever have to think about. A native integration would handle this transparently as part of initialisation: detect the `.devcontainer`, spin it up, manage the lifecycle. No instructions required.

It's a reasonable stopgap. But it's also exactly the kind of manual wiring that shouldn't need to exist.

## Others are reinventing the wheel

The community has noticed the problem, but the responses so far have all approached it from the wrong angle.

Docker Sandboxes[^1], launched in January 2026, provides a microVM-based sandbox for Claude Code to run in. It solves safety and isolation — and Docker-in-Docker — but it doesn't solve the reproducible environment problem at all. It ignores your devcontainer entirely and gives you a fresh sandbox instead. Arcade did a hands-on review[^2] that captures the core issue well: setup is smooth for simple tasks, but falls apart fast with real workflows — missing binaries, environment parity issues, and API keys requiring a full sandbox restart that wipes your Claude session. Their conclusion: execution isolation and environment parity are very different problems. Docker Sandboxes solves the first and ignores the second. Your devcontainer is already the answer to the second.

Before Docker Sandboxes, Kevin Scott wrote up his DIY approach[^3] — essentially hand-rolling what Docker Sandboxes later productised. He runs Claude Code inside a heavily customised Docker container based on Anthropic's own `.devcontainer` example, with pre-installed tooling, a dedicated GitHub identity for the agent, and tmux + git worktrees to run multiple Claude instances in parallel. It's impressive, and it works, but it's also a lot of manual setup. The irony is that his solution is basically a hand-rolled devcontainer.

## Claude needs to make friends with devcontainers

Both approaches are workarounds for a missing abstraction.

The devcontainer spec was never really about VS Code. The spec itself is a _project-level environment definition_. It describes what your project needs to run: the base image, installed tools, environment variables, port mappings, lifecycle hooks.

In my opinion, that definition is equally useful to a human developer and an AI agent. Right now, VS Code is the primary consumer and Claude Code isn't. However, if it were, the model becomes quite simple:

- human opens VS Code → devcontainer spins up;
- Claude Code runs → detects `.devcontainer` → same container.

One spec, two consumers. No duplication, no drift, no manual configuration.

The containerisation problem is already solved. The only missing piece is the link from Claude to the container.

## Cloud environments come for free

If the above is a convenience argument, this is the one that makes native devcontainer support feel necessary rather than nice-to-have.

If you want to allow Claude Code to run autonomously in the cloud, then you need to define an execution environment. Right now, configuring that environment means writing a bespoke script or maintaining a separate Docker image — which immediately starts to drift from your local devcontainer.

This is the same problem GitHub Codespaces already solved for human developers. Define your `.devcontainer.json` once, and you get a fully reproducible cloud environment on demand with no extra configuration. The devcontainer spec was always designed with this in mind; the cloud execution model isn't an afterthought, it's a first-class use case.

With native devcontainer support, Claude Code cloud tasks would work the same way. Your `.devcontainer.json` _is_ the cloud environment spec. Already defined, already tested locally. You don't write anything new; you get it for free. And because it relies on Docker, container builds are fast due to caching. The same logic applies to Claude Code web, which similarly needs a defined environment to operate in.

{{< alert icon="lightbulb" >}}
One spec, four consumers: you locally, you in Codespaces, Claude locally, Claude in the cloud.
{{< /alert >}}

## This is Anthropic's problem to fix

The fix is actually really simple: Claude Code detecting a `.devcontainer` in the repo root and routing execution through it. The only extra challenge is managing the container lifecycle, but that's surmountable.

This is similar to how MCP solved the "how do tools talk to agents" problem. There was already a proliferation of one-off integrations; MCP gave them a standard shape. Right now, everyone building Claude Code workflows with containers is fumbling toward their own solution. The devcontainer spec already exists and is widely adopted. Claude Code just needs to consume it.

I checked the Claude Code GitHub issues recently. Surprisingly, nobody had raised this yet — so I did[^4]. If you agree, add a comment to the issue.

For now, I'm using the command prefix approach locally which _mostly_ works. But I'm aware it's held together with string, and I'm not sure how it scales to cloud tasks at all.

The right answer is native devcontainer support. The spec exists. Anthropic just needs to bridge the gap.

Having built the workaround, I'm more convinced of this than ever. The fact that Claude doesn't need to know anything about the container — the hook handles all the routing transparently — shows how narrow the actual integration surface is.

{{< alert icon="lightbulb" >}}
If a rough PoC built in an afternoon gets you most of the way there, a proper native integration would be essentially invisible. That's exactly the bar Anthropic should be aiming for.
{{< /alert >}}

[^1]: [Docker Sandboxes announcement](https://www.docker.com/blog/docker-sandboxes-run-claude-code-and-other-coding-agents-unsupervised-but-safely/)
[^2]: [Arcade hands-on review of Docker Sandboxes](https://www.arcade.dev/blog/using-docker-sandboxes-with-claude-code/)
[^3]: [Kevin Scott's DIY Claude Code sandbox](https://thekevinscott.com/sandbox-for-claude-code/)
[^4]: [GitHub issue: native devcontainer support for Claude Code](https://github.com/anthropics/claude-code/issues/47856)
