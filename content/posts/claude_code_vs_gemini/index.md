+++
title = 'Tried to save £18/month on Claude Code, wasted £5 in one evening on Gemini'
date = 2025-03-27T23:06:00Z
tags = ['development']
+++

I recently set up a new homelab; an old Dell PC running Ubuntu server. Multiple networking and permissions issues to debug. I thought an AI coding agent would massively speed this up.

Claude Code is £18/month. For hobby use, that felt expensive. I assumed Gemini would work just as well for less money. I had access via a Google AI trial, so why not?

**Spoiler: I was wrong on multiple levels.**

## Tier 1: Gemini CLI works fine... for simple tasks

To be fair, raw Gemini CLI actually works well for straightforward conversions. When I needed to [convert my LaTeX CV to Typst]({{< ref "/posts/typst" >}}), it handled it without issues.

I thought: "If it can handle that, homelab debugging should be fine too."

The real test came when I tried to set up a Samba share for Time Machine backups. I needed:

- Modern Samba version (more efficient streaming for virtual APFS)
- Configurable Docker image (to build a runtipi app later)

Gemini found two Docker images:

- [dperson/samba](https://github.com/dperson/samba) (old version)
- [crazy-max/docker-samba](https://github.com/crazy-max/docker-samba) (new, but hard to configure)

I explicitly told it: "I need modern Samba." This clearly rules out dperson.

**It kept suggesting dperson anyway.** Over and over. Completely ignored my constraint.

It also kept getting stuck in infinite loops, where I had to intervene.

## Tier 2: Adding a harness didn't help

I thought that this was because the Gemini CLI harness wasn't very good. So tried using [OpenCode](https://opencode.ai) (an agentic harness) with Gemini under the hood.

What initially drew me to it was that it's a genuinely sophisticated product. Unlike raw CLI tools, it has a web GUI so you can code from the browser without SSH, a dedicated thinking mode, and you can even configure different models for different roles — a cheaper model for small edits, a more powerful one for planning. On paper, it seemed like a step up in every way.

### New harness, same results

Unfortunately, despite the fancy UI, the experience was very similar to Gemini CLI. It kept suggesting the "infeasible solution", except that it did give up and explain why it was stuck.

Here's the key insight: **adding a more sophisticated harness made no difference.** That doesn't mean the harness is irrelevant — it might well be making things worse in ways that are hard to observe. But it's clearly not enough to compensate for a weak underlying model.

### The API cost shock

I thought maybe Gemini Flash was just too cheap. Let me try Gemini Pro via API.

I spent roughly **£5 in one evening** across various homelab debugging tasks (not just the Samba issue).

That's 25% of a monthly Claude Code subscription. In one evening.

## Tier 3: Claude Code actually solved it

At this point, I realised I should probably try what I know works: Claude Code.

**Claude's approach to the same Samba problem:**

- Planned before jumping in; actually understood what I was trying to achieve
- Recognised that the two images I'd found weren't a good fit, and went looking for alternatives
- Found [mbentley/docker-timemachine](https://github.com/mbentley/docker-timemachine) — a purpose-built image I hadn't come across
- SSH'd into the homelab and tested from both sides
- **Actually solved it**

The rule of thumb is that for low usage, PAYG is cheaper. So I initially tried Claude via API, thinking it would be cheaper than the subscription. Whilst I did solve my problem, I spent another ~£5 in the process.

After this, I bit the bullet on Claude Pro (£18/month). You get loads of usage plus nice-to-haves like remote control feature (can spin up tasks from my phone), Claude Pro chat bundled in. It's a proper product, not just API access.

## The three lessons

### Lesson 1: A better harness can't fix a weaker model

Gemini CLI is fine for simple conversions like [LaTeX to Typst]({{< ref "/posts/typst" >}}). However, it struggles severely when solving more complex tasks.

There's clearly a fundamental model limitation, and not _just_ a tooling issue. Gemini kept taking shortcuts and ignoring my constraints, even with a better agentic harness.

### Lesson 2: Claude still has an advantage _right now_

Right now, you can't get the same experience for less money. The quality gap is real for systems work; it's not just about features, it's about problem understanding, reasoning, and finding the right solutions.

That said, this space is moving fast. OpenCode with a stronger model choice might close the gap — I only tested it with Gemini. And the model landscape in 6 months will look very different. It's worth keeping an eye on.

### Lesson 3: The subscription threshold is lower than you think

This goes against the conventional rule of thumb that API pricing is cheaper for light usage.

I spent **£5 in one evening** of hobbyist debugging. At that rate, the £18/month subscription pays for itself after just 4 evenings of work; and that's not counting the time lost fighting with Gemini.

Developer time is expensive (even if it is just for your hobbies!). The subscription makes sense much earlier than I'd expected.
