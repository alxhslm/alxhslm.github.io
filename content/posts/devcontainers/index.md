+++
title = 'Dev containers are awesome'
date = 2024-03-01T18:35:54Z
draft = true
tags = ['development']
+++

I personally love finding ways to reduce friction in my life easier and optimise my processes [^1]. The best tools are the ones which fit in easily to your workflow, and make you wonder how you worked without them before. Dev containers are one of those tools for me, and I’m going to explain why I like them so much.

## What are dev containers?

Dev containers are special Docker containers where you develop your code. This means that you just need Docker installed on your host machine, and then all dependencies are installed within the container.

Dev containers were popularised by VS code, but there is in fact an [open specification](https://containers.dev/). Dev containers can be used with multiple editors (incl. full-fat Visual Studio and IntelliJ based IDEs) and various cloud services (eg. GitHub Codespaces).

You define the environment through a `Dockerfile` (and possibly a `docker-compose.yaml`) and a configuration `devcontainer.json` file. These files configure the container for your project including installing any dependencies, and can be stored in your Git repo along with your source code.

## Why do I need a dev container?

### Zero-setup

The best-case scenario is when someone _else_ has done the hard-work and defined the dev container for you [^2]. There’s no need to read “Get started” instructions for a new project, you just:

1. Pull the repo
2. Launch VS code
3. Start developing

This is particularly beneficial when your project has **complex system requirements.** One person works out the fiddly bits of how to set things up, and then shares it with everyone; either you pull the updated image, or rebuilding their container.

This proved invaluable at some of my previous company once when we started to build bits in different languages. I was able to get developing with a whole new eco-system of tools right away, without having to waste time configuring things.

You might wonder why this matters if you don’t work on many new projects and the system dependencies are stable. However, even in this scenario, dev containers can be helpful. For example, what happens if your machine breaks and you want to get started using a new one as quickly as possible?

### Reproducibility

Dev containers help to fix a lot of “Works on my machine” problems. It can happen that the system dependencies to have changed since you were last working on your branch, and then you find your code no longer works. Without a dev container, you may end up spending a lot of time debugging before you _eventually_ realise that there was some mismatch in the system dependencies between your main and development branches.

On the other hand, when you use dev containers, you could just rebuild the old version of the container to work on the old branch to carry on developing. And if there is a **conflict** in system dependencies, this should become apparent when you merge in your main branch.

### Isolation

The other major benefit of dev containers is that your dependencies are completely isolated from your host machine. You can't accidentally break your machine - if you make a mistake in the dev container definition, you can just modify your `Dockerfile` and rebuild the container. It's also trivial to completely remove all dependencies for a given project.

Your projects are also completely isolated from each other, so that you never run into issues with conflicting system dependencies. I personally used to find that when developing locally, I would waste time trying to debug an issue before I realised I accidentally activated the wrong Python virtual environment. This is very unlikely to happen when using a dev container because it would (likely) only contain a single virtual environment.

## Configuring a dev container

The first time you set things up, it can be quite tricky. If you’re working on a project with a small team and not many system dependencies, then the benefits of a dev container will be limited. This means that it’s harder to justify the upfront investment in time.

However, there's no need to start from scratch. You can:

- Make use of [pre-made dev container templates](https://containers.dev/templates) which you can then build on top of
- Add [pre-made “features”](https://containers.dev/features) to your definition, which carry out common tasks for you such as installing the AWS CLI

In my experience, it becomes a lot easier to set up a dev container the second time around, because you can often copy a lot of the dev container definition from a previous project.

## Using a dev container in the cloud

A nice bonus of containerising your development environment is that you can develop on _any_ host machine, including ones in the cloud. The most obvious example is [GitHub Codespaces](https://github.com/features/codespaces), where you can spin up a new environment from a branch right from the GitHub UI. There are other cloud development providers such as [Gitpod](https://www.gitpod.io/), [Coder](https://coder.com/) and [CodeSandbox](https://codesandbox.io/). I use Codespaces for personal projects quite a lot, since my machine is quite old and sometimes struggles, but you could even use this approach to develop on an iPad.

A tool I have started to use recently is [DevPod](https://devpod.sh/). This is an open-source tool which you install locally on your machine, and can then be used to spin up cloud dev environments _almost_ as seamlessly as GitHub Codespaces. The key difference is that you can use **any** cloud provider such as AWS or GCP. This can work out to be much cheaper if your usage is high[^3], and allows you to further customise the host machine (eg adding GPUs).

[^1]: Or possibly even over-optimise? 😅
[^2]: Even better, they may have even built an image for you
[^3]: You can read an interesting price comparison between AWS and Codespaces [here](https://pauley.me/post/2022/ec2-codespace-autostart/)
