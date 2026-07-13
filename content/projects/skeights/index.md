+++
title = 'skeights'
date = 2026-07-03T00:00:00Z
tags = ['machine learning', 'mlops', 'python', 'open source']
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
opaque. However, the biggest problem was that we build wrapper
classes around the sklearn estimators to add extra functionality
we need, and these were also included in the pickle. This meant
that any refactor of our code could break previously saved
models, so backwards compatibility was a constant pain. This
wasn't workable in the long term.

## Existing tools don't cut it

Hugging Face make [`skops`](https://github.com/skops-dev/skops),
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

## Rolling our own

We ended up solving this problem internally, by building our own
serialization that decomposes a fitted estimator into JSON
(hyperparameters) and safetensors (weights). Over time it grew
to cover most of the sklearn ecosystem, including `lightgbm` and
`xgboost`. Those two were the easy part, since they already
serialise to JSON natively; most of the effort went into the
sklearn estimators, which don't.

At some point I realised this was a generic serialisation layer
which could be useful to others. The serialization code already
had no dependency on our domain code, so it was quick and easy
to extract.

The result is `skeights`[^1]:

{{< github repo="carbon-re/skeights" >}}

It exposes a simple API to save and load models:

```python
import skeights

skeights.save(fitted_pipeline, "model.safetensors", "model.json")
loaded = skeights.load("model.safetensors", "model.json")
```

It works out of the box with the sklearn-like estimators most
people use, which is what we've covered so far.

## A worked example

Say you train a simple pipeline; a Ridge regression with some
scaling on top:

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import skeights

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=0.1)),
])
pipe.fit(X_train, y_train)

skeights.save(pipe, "model.safetensors", "model.json")
```

The hyperparameters and structure end up in `model.json`, which
you can read at a glance:

```json
{
  "model_params": {
    "steps": {
      "scaler": {
        "copy": true,
        "with_mean": true,
        "with_std": true,
        "type": "sklearn.preprocessing.StandardScaler"
      },
      "model": {
        "alpha": 0.1,
        "fit_intercept": true,
        "solver": "auto",
        "type": "sklearn.linear_model.Ridge"
      }
    },
    "type": "sklearn.pipeline.Pipeline"
  }
}
```

while the fitted arrays go into `model.safetensors`:

```
scaler/mean_       (3,)  float64
scaler/scale_      (3,)  float64
model/coef_        (3,)  float64
model/intercept_   ()    float64
```

## Models as config

The major advantage is that a model is now (mostly) JSON, so you can
build and manipulate models as data rather than code. A
hyperparameter sweep becomes a matter of generating configs
instead of instantiating estimators in Python.

It also composes nicely. We save the model config as just one
field within a larger JSON object alongside the rest of our
config, rather than having to handle a separate binary artefact
on the side.

## Columnar tree serialization

While `lightgbm` and `xgboost` natively serialize to text or JSON, storing the entire model structure this way has drawbacks. The resulting files are huge and unreadable, defeating the benefit of having a human-readable config. For example, a native `xgboost` JSON export contains a massive string representing all of its trees:

```json
{
  "model_params": {
    "type": "xgboost.XGBClassifier",
    "raw_model": "{\"learner\":{\"gradient_booster\":{\"model\":{\"trees\":[{\"id\":0,\"split_indices\":[1,0,0],\"split_conditions\":[0.5,1.1,-0.2],\"left_children\":[1,-1,-1],\"right_children\":[2,-1,-1]...[hundreds of thousands of characters truncated]...\""
  }
}
```

Instead of using this native text format, `skeights` extracts the tree parameters (such as split features, thresholds, child pointers, and leaf values) and saves them as typed numpy arrays in the `safetensors` file. Only the high-level scalar configuration remains in the JSON:

```json
{
  "model_params": {
    "type": "xgboost.XGBClassifier",
    "objective": "binary:logistic",
    "num_class": 1,
    "feature_names": ["f0", "f1", "f2"]
  }
}
```

The corresponding `safetensors` file contains the arrays:

```
model/trees/split_indices      (1200,)  int32
model/trees/split_conditions   (1200,)  float32
model/trees/left_children      (1200,)  int32
model/trees/right_children     (1200,)  int32
```

This reduces file sizes significantly compared to native formats or pickle. For a model with 100 trees and 20 features:

- `xgboost`: 85% total reduction, with the JSON dropping from 1.8MB to 19KB.
- `lightgbm`: 30-43% total reduction, with the JSON dropping from 129-193KB to 6KB.

The native format is still available as a fallback by passing `format="native"`, and legacy artifacts load transparently.

## Security

By using JSON and safetensors, `skeights` avoids the arbitrary code execution risks of pickle out of the box.

However, since the loader still has to instantiate Python classes specified in the JSON config, a crafted JSON file could try to import arbitrary modules. To prevent this, the loader restricts class imports to an allowlist containing only `sklearn`, `lightgbm`, and `xgboost` modules. Attempting to load a file with unauthorized module paths raises an error, strengthening the overall security model.

## Drawbacks

We have to add support for each estimator by hand, so we only
cover the ones we've done so far. Backwards compatibility is also
tricky as the library grows. We have tests that aim to catch
breakages, but it isn't guaranteed; then again, neither pickle
nor `skops` guarantee this either.

## Try it out

We're now using `skeights` in production. It's MIT licensed and you
can install it with `pip install skeights`. Contributions welcome,
especially to widen the models we support.

[^1]: Pronounced "skates".
