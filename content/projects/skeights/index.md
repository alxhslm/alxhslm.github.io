+++
title = 'skeights'
date = 2026-07-03T00:00:00Z
tags = ['machine learning', 'python', 'open source']
summary = 'Extracting and open-sourcing our sklearn serialization library.'
+++

At work we build various sklearn models in production to solve
time series problems in industrial settings. These models get
trained, versioned, and deployed across different environments.
The standard way to save a fitted sklearn model is pickle, but
this is far from ideal.

There is the obvious security concern: loading a pickle executes
arbitrary code. It is also not stable across sklearn versions.
You can't see what is inside the model as the binary blob is
opaque. However, the biggest problem though was that we build wrapper
classes around the sklearn estimators to add extra functionality
we need, and these were also included in the pickle. This meant
that any refactor of our code could break previously saved
models, so backwards compatibility was a constant pain. This
wasn't workable in the long term.

Hugging Face make [skops](https://github.com/skops-dev/skops),
an existing library for saving sklearn models in a safe binary
format. This solves the security issue but the output is still
an opaque blob.

What's ironic is that PyTorch doesn't have this problem, even
though the models are far more complex. `state_dict()` gives you
a clean dict of tensors, and safetensors handles the rest.
Separating weights from structure is part of PyTorch's design,
but unfortunately this is not the case for sklearn. The fitted
state is scattered across various private attributes with no
standard way to extract or restore it.

We ended up solving this problem internally, by building our own
serialization that decomposes a fitted estimator into JSON
(hyperparameters) and safetensors (weights). Over time it grew
to cover most of the sklearn ecosystem, including LightGBM and
XGBoost.

At some point we realised this was a generic serialisation layer
which could be useful to others. The serialization code already
had no dependency on our domain code, so it was quick and easy
to extract.

The result is skeights[^1]:

{{< github repo="carbon-re/skeights" >}}

```python
import skeights

skeights.save(fitted_pipeline, "model.safetensors", "model.json")
loaded = skeights.load("model.safetensors", "model.json")
```

It works out of the box with most common sklearn-like estimators.

The major bonus of this is that model configuration is now
structured data. Hyperparameter sweeps become config-driven.
Agents can inspect, modify, and create models very easily as
it's all plaintext. When your model state is plain JSON, it is much easier
to build useful tooling on top of it.

We're now using skeights in production. It's easy to install with
pip, and is MIT licensed. Contributions welcome, especially to
widen the models we support.

[^1]: Pronounced "skates".
