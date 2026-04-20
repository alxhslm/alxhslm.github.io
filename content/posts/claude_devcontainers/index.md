+++
title = 'Claude Code and devcontainers: the missing link'
date = 2026-04-17T21:00:00Z
tags = ['tooling', 'devops', 'agents']
+++

I've [written before]({{< ref "/posts/devcontainers" >}}) about how much I like devcontainers; they make your environment completely reproducible and keep projects isolated from each other.

However, a new actor called Claude has entered the picture, which doesn't fit neatly into the model devcontainers were designed around.

## The two-body problem

Devcontainers were built around one mental model: _everything_ executes inside the container. VS Code attaches to it, your terminal runs inside it, your tools are installed inside it.

Claude Code has the opposite assumption. It's an agent running on your machine, shelling out commands directly. It expects to be able to reach your filesystem, your credentials, your Docker daemon. It was designed to operate _alongside_ your environment, not inside it.

These two mental models conflict. Claude Code has no awareness of devcontainers: it won't detect a `.devcontainer` in your repo, and it won't use one if it exists. I've been experimenting with approaches to bridge that gap.

### Others are reinventing the wheel

The community has noticed the problem, but I don't think anyone has found the right solution yet. Kevin Scott first wrote up a DIY approach[^1] running Claude Code inside a heavily customised Docker container with pre-installed tooling, a dedicated GitHub identity, and `tmux` + `git worktrees` for parallelism.

Docker Sandboxes[^2] later productised this idea, providing a microVM-based sandbox for Claude Code to run in. Both solve safety and isolation, but neither solves the reproducible environment problem. They ignore your devcontainer entirely and give you a fresh sandbox instead. Arcade did a hands-on review[^3] that captures the core issue well: setup is smooth for simple tasks, but falls apart fast with real workflows. For example, a sandbox restart wipes your Claude session. The irony is that both solutions are essentially hand-rolled devcontainers.

## Finding my own solution

### Approach 1: install Claude inside the container

This was the obvious starting point. If Claude needs to run commands in the container, just put Claude in the container.

This was very easy to implement, but is more painful than it sounds:

- **Session history breaks.** Claude Code keys session history to the host path. If the path inside the container differs from the host (say, `/home/alex/project` vs `/workspace/project`), Claude doesn't know they're for the same project. This means a bind mount of the `.claude` directory is not sufficient.
- **Credentials don't transfer.** Currently you can't pass Claude credentials from the host to the container like you can with Git credentials. This means you need to reauthenticate every time the container is rebuilt.
- **It pollutes the project environment.** Claude Code has nothing to do with your project's actual dependencies, but it ends up baked into every contributor's container. Not everyone uses Claude Code, and those who do shouldn't need to maintain it as part of the project image.
- **Docker access is limited.** Running `docker build` from inside a container means Docker-in-Docker (fragile) or Docker-from-Docker (better, but needs to be set up). In a typical `docker-compose` setup (app, database, cache), Claude can only reach the container it's running in. Claude on the host can reach all of them.

Running Claude inside the container works, but it's not a particularly smooth setup. It's the wrong level of abstraction: the devcontainer should describe your project, not your AI assistant.

### Approach 2: get Claude on the host to use the container

The alternative is to keep Claude on the host but make it execute tasks inside the container's environment. This immediately solves the issues with credentials, session history, and Docker access. The question is whether you can route Claude's commands through the container from the host.

It turns out that Claude makes this quite easy with [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks-guide), which let you intercept tool calls including bash commands. By prefixing every shell command with `docker exec -it <container>`, Claude runs its commands inside the container while staying on the host.

I built a proof of concept at work that works by registering a Claude hook that forwards shell commands into the running container. You can see the example repo below.

{{< github repo="alxhslm/claude-devcontainer" >}}

What surprised me was how seamlessly it works in practice. Because it's a hook rather than a skill or prompt instruction, Claude cannot ignore it: every shell command is guaranteed to be routed through the container. This means Claude can interact with the container's installed tooling (like running unit tests), while retaining access to general tools on the host like `gh` CLI or the Docker daemon.

The one genuinely unsolved piece is the container lifecycle. The devcontainer only spins up when VS Code opens it. I handled this through the `CLAUDE.md` file: Claude is instructed to check whether the container is running at the start of a session and ask the user to start it if not, but it's a bit brittle. A native integration could handle this transparently as part of initialisation: detect the `.devcontainer`, spin it up, manage the lifecycle.

## Claude needs to make friends with devcontainers

The devcontainer spec was never really about VS Code. It's a _project-level environment definition_. It describes what your project needs to run: the base image, installed tools, environment variables, port mappings, lifecycle hooks.

In my opinion, that definition applies equally well to a human developer and an AI agent. Right now, VS Code is the primary consumer and Claude Code isn't. However, if it were, the model becomes quite simple:

- human opens VS Code → devcontainer spins up;
- Claude Code runs → detects `.devcontainer` → uses the same container.

One spec, two consumers. No duplication, no drift, no manual configuration.

The containerisation problem is already solved. The only missing piece is the link from Claude to the container.

## Cloud environments come for free

The previous argument is about convenience. This one makes native devcontainer support feel necessary rather than nice-to-have.

If you want to allow Claude Code to run autonomously in the cloud, then you need to define an execution environment. Right now, configuring that environment means writing a bespoke script or maintaining a separate Docker image with no relation to your devcontainer.

GitHub Codespaces already solved this problem for human developers. Once you define your `.devcontainer.json`, you can get a fully reproducible cloud environment in Codespaces with no extra configuration. With native devcontainer support, Claude Code cloud tasks would work the same way. Your `.devcontainer.json` _is_ the cloud environment spec.

Since devcontainers relies on Docker, container builds are fast due to caching. Your local and cloud environments are kept in sync for both humans and agents.

{{< alert icon="lightbulb" >}}
**One spec, four consumers:** VS Code, Codespaces, Claude Code (local), Claude Code (cloud).
{{< /alert >}}

## This is Anthropic's problem to fix

The fix is actually really simple. Claude Code just needs to detect a `.devcontainer` in the repo root and route execution through it. I've already demonstrated it can work in principle. The only extra challenge is managing the container lifecycle, but that's trivial for a company like Anthropic.

They've already solved the "how do tools talk to agents" problem with MCP. Right now, everyone building Claude Code workflows with containers is fumbling toward their own solution, but the devcontainer spec already exists and is widely adopted. Claude Code just needs to consume it.

For now, I'm using the command prefix approach locally which _mostly_ works. However, the right answer is for Anthropic to add native devcontainer support. I've added this Claude Code issue on GitHub[^4]. If you agree, please show your support by adding a comment to the issue!

[^1]: [Kevin Scott's DIY Claude Code sandbox](https://thekevinscott.com/sandbox-for-claude-code/)
[^2]: [Docker Sandboxes announcement](https://www.docker.com/blog/docker-sandboxes-run-claude-code-and-other-coding-agents-unsupervised-but-safely/)
[^3]: [Arcade hands-on review of Docker Sandboxes](https://www.arcade.dev/blog/using-docker-sandboxes-with-claude-code/)
[^4]: [GitHub issue: native devcontainer support for Claude Code](https://github.com/anthropics/claude-code/issues/47856)
