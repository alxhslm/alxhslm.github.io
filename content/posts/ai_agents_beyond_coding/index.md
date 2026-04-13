+++
title = 'Learning to Let Go: Using AI Agents Beyond Coding'
date = 2026-04-12T20:00:00Z
tags = ['ml', 'homelab', 'agents']
+++

I recently set up an SMB share for Time Machine backups on my homelab. It's a fiddly task fighting networking and permissions issues, and Docker configuration. I thought an AI agent would help speed things up.

It turned out to echo something I keep running into at work with ML models: to get the best out of an agent, you need to relinquish control in ways that feel uncomfortable, but only with the right guardrails in place.

## Setting up Time Machine over SMB

I wanted to set up an SMB network share on my Ubuntu homelab to use as a Time Machine backup destination[^1]. The requirements were:

- Containerised within Docker
- Configurable enough to build a runtipi app later
- Modern Samba version (more efficient streaming for virtual APFS)

**Definition of done:** The Mac can see the SMB share AND successfully read and write to it.

### The naive approach: using myself as a middleman

Initially, I ran Claude Code on the homelab server and told it to set up the share. Then I'd manually test the connection from my Mac and report back any errors.

This was slow. I had to:

- Describe what wasn't working with enough context
- Remember to test all functionality (I kept forgetting to check write access)
- Go back and forth between machines

I knew the agent could test the connection itself, and that it could SSH between the two machines. But I kept doing it manually anyway. Partly because I'm used to being hands-on, but more fundamentally, it felt risky to give an agent full SSH access to both machines for something this simple; like using a sledgehammer to crack a nut.

### The shift: let the agent own everything

Then I realised: **why am I doing half the work when the agent can do all of it?**

I changed the workflow:

1. Run Claude Code on my Mac (not the server)
2. Have it SSH into the homelab server
3. Now it can configure the server AND test the connection from the Mac
4. It sees full error context from both sides and can iterate on its own

The agent would:

- SSH in, adjust the `smb.conf`
- Test the mount from the Mac side
- See the error message
- SSH back in, fix the config
- Repeat until it worked

Then I just watched it debug itself. It was much faster.

However, **this only worked because I had tight constraints beforehand.** I'd explicitly told it: modern Samba, must be containerised, needs to work with runtipi. Without those guardrails, it would have taken shortcuts that technically worked but didn't meet my actual requirements.

## Building models is similar

At work, I've started to use AI agents to iterate on machine learning models for time-series forecasting, and I've noticed the same pattern.

If you give the agent free rein to optimise a forecasting model without a well-defined evaluation methodology, it will confidently report unbelievable metrics. This is because the agent can cheat.

Data leakage in time-series is a classic failure mode. The model uses future data to predict the past, results look amazing, but they're completely useless. The agent doesn't know this is wrong because it just sees the metrics improving.

{{< alert icon="lightbulb" >}}
**What I've learnt:** Unrestricted freedom without guardrails can give bad results.
{{< /alert >}}

For ML work, the guardrail is evaluation. Proper train/test splits, validation that catches temporal leakage, metrics that actually measure what you care about. Once you've defined that, the agent can iterate through dozens of model configurations in minutes.

For sysadmin work, the guardrail is end-to-end testing. Not "config looks right", but "does it actually work when I try to use it?"

## The pattern

I think relinquishing control is psychologically hard, especially for technical people who are used to being hands-on. But the agent is 10x faster at iteration than we'll ever be.

The key is knowing where our judgement still matters:

- **Less valuable:** Knowing specific `smb.conf` or PyTorch syntax, because the agent can do this well already.
- **More valuable:** Defining what "correct" looks like, and how to test for it, because this is where the agent can trip up.

{{< alert icon="lightbulb" >}}
The paradox I've found: I need to be **more** rigorous about defining success precisely so I can be hands-off about how it gets achieved.
{{< /alert >}}

For me, this has meant:

1. **Defining requirements clearly beforehand**: otherwise the agent takes shortcuts around them
2. **Making "done" testable**: giving the agent something it can verify itself
3. **Giving it access to test its own work**: getting myself out of the iteration loop
4. **Intervening when it gets stuck**: asking it to summarise root cause and suggest alternatives

## Where I think domain expertise matters

I think that our role as engineers has shifted; it's more about setting up the problem so the agent can explore it effectively, rather than doing the manual iteration ourselves.

For time-series forecasting: "Does this evaluation catch the ways my model could be wrong?"

For SMB configuration: "Does this test prove the system actually works end-to-end?"

For any technical work: "What are the failure modes I care about, and how do I detect them?"

If you define it correctly, the agent can move fast. If not, we're just automating the production of plausible-looking slop.

I'm still not entirely sure what the best way to enforce these guardrails is. Is it through writing skills that the agent can reference? Is it just careful prompting? I suspect it's a mix of both, but I haven't figured out the right balance yet.

[^1]: I also wrote about [my issues with Gemini on this task]({{< ref "/posts/claude_code_vs_gemini" >}}).
