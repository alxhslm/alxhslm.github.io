+++
title = 'Coffee Rating Prediction: Modelling'
series = ['Coffee Rating Prediction']
series_order = 2
date = 2023-11-13T00:00:00Z
showTableOfContents = true
+++

This was my first personal ML project. The goal was to familiarise myself with [XGBoost](https://xgboost.readthedocs.io/en/stable/) and [AWS Lambda](https://aws.amazon.com/lambda/). I enjoy drinking nice coffee, so I chose a topic which I hoped would help me buy better coffee in the future. The source code is available on GitHub.

{{< github repo="alxhslm/coffee-rating-prediction" >}}

## Background

In a [previous post]({{< ref "../eda" >}}) in this series, we performed experimental data analysis to understand the distribution of the rating and get some insight as to which features might be important.

The next step is to use this information to develop and train a model on the data which is capable of predicting the ratings.

## Feature engineering

### Roaster

Let's first verify that the information about each roaster (in this case only country) is consistent across all coffees from the same roaster.

```python
def _assert_identical_values(df: pd.DataFrame) -> pd.Series:
    assert (df.iloc[1:, :] == df.iloc[0, :]).all().all()
    return df.iloc[0, :]


roaster_map = df[["roaster", "roaster_country"]].groupby("roaster").first()["roaster_country"]
```

Since the previous analysis above showed that there is evidence that certain roasters product particularly good/poor coffee (or are preferred/disliked by the reviewers), the model may therefore need a feature giving it this information. We cannot simply convert the roaster using one-hot encoding as there are too many different values. Let us instead only include the most common roasters (those with > 10 coffees).

```python
roasters = df["roaster"].value_counts()
popular_roasters = sorted(roasters[roasters > 10].index)
```

Let's save this information to [`roasters.json`](https://github.com/alxhslm/coffee-rating-prediction/blob/main/data/roasters.json) for later use.

```python
roaster_info = {"known_roasters": roaster_map.to_dict(), "popular_roasters": popular_roasters}
with open("data/roasters.json", "w") as f:
    json.dump(roaster_info, f, indent=4)
```

We can now use this information to engineer roaster features with a smaller number of uniqute values.

```python
df["roaster"] = df["roaster"].where(df["roaster"].apply(lambda r: r in popular_roasters), "Other")
```

{{< alert >}}
If we find these features have a strong influence on the model, we need to be careful when applying the model to new coffees from unknown roasters. Even if the coffee is from a well-known roaster, they will have a "roaster" value of "Other" if they are not present in the training set.
{{< /alert >}}

### Region of origin

The different regions of the world typically produce coffees which are similar in style. Eg African coffees are typically more acidic. Therefore it seems possible that the region may provide as much information as the country of origin. We will therefore engineer this feature.

```python
with open("./data/regions.json", "r") as f:
    REGIONS = json.load(f)

regions = {}
for r, countries in REGIONS.items():
    for c in countries:
        regions[c] = r

df["region_of_origin"] = df["country_of_origin"].map(regions).fillna("Other")
```

The vast majority of coffees in the dataset come from the major coffee producing regions of the world as expected.

```python
df["region_of_origin"].hist(histnorm="percent", labels={"value":"region of origin"})
```

{{< include src="charts/origin_region_hist.html" >}}

### Flavour notes

As it stands, we cannot glean any information from the review column as it is unstructured. Let's begin by analysing the keywords present in the reviews.

```python
import re


def extract_words(string: str) -> list[str]:
    return re.findall(r'\w+', string.lower())


words = pd.Series([word for review in df["review"] for word in extract_words(review)]).value_counts()

GENERIC_WORDS = ["and", "in", "with", "the", "of", "to", "a", "by", "like", "is", "around"]
COFFEE_WORDS = ["cup", "notes", "finish", "aroma", "hint", "undertones", "resonant", "high", "consolidates", "flavor"]
words = words.drop(GENERIC_WORDS + COFFEE_WORDS)
words.head()
```

We can see that the most common words relate to the flavour of the coffee. This suggests that we can extract some features for the different flavours in the coffee.

Using this information and the [coffee flavour wheel](https://www.anychart.com/products/anychart/gallery/Sunburst_Charts/Coffee_Flavour_Wheel.php), we can manually define some flavours and corresponding keywords which are stored in [`flavours.json`](https://github.com/alxhslm/coffee-rating-prediction/blob/main/data/flavours.json).

```python
with open("./data/flavours.json", "r") as f:
    FLAVOURS = json.load(f)
```

We can now engineer boolean features for each flavour.

```python
def rating_contains_words(review: str, keywords: list[str]) -> bool:
    words = extract_words(review)
    for w in keywords:
        if w in words:
            return True
    return False


for flavour, keywords in FLAVOURS.items():
    df[flavour] = df["review"].apply(rating_contains_words, args=(keywords,))
```

Let's now combine the flavours into a single column.

```python
df["flavours"] = df.apply(lambda coffee: [flavour for flavour in FLAVOURS if coffee[flavour]], axis=1)
```

It is useful to examine the popularity of the different flavours, by plotting the histogram.

```python
df[list(FLAVOURS.keys())].sum().divide(df.shape[0]).sort_values(ascending=False).plot.bar(labels={"index":"flavour"})
```

{{< include src="charts/flavour_hist.html" >}}

We can see that the most common flavours are:

- Caramelly
- Acidic
- Fruity
- Chocolate

Intuitively, this makes sense as these are the sorts of flavours you commonly see mentioned on packets of coffee.

It is also interesting to check how many flavours the different coffees have. If we have done a good job at defining the flavour keywords, we would expect:

- Most coffees to have at least some flavours since this is a key component of any review
- Most coffees to not have an excessive number of flavours, as this would indicate we have chosen too "common" keywords

```python
df[list(FLAVOURS.keys())].sum(axis=1).hist(histnorm="percent", labels={"value":"num_flavours"})
```

{{< include src="charts/num_flavours_hist.html" >}}

Indeed, this appears to be the case. All coffees have at least some flavours, and in fact most coffees have ~6 flavours.

## Building a model

We begin by selecting the features to use. In addition to the raw "roast", "roaster_country" and "price_per_100g" columns, we will use the following engineered features:

- "roaster" where we have encoded the most highly and poorly rated roasters
- "region_of_origin" where we have categorised the country of origin into major coffee producing regions
- "flavours" where were have extracted flavour keywords from the reviews

```python
features = ["roaster", "roast", "roaster_country", "region_of_origin", "price_per_100g", "flavours"]
X = df[features].copy()
X["price_per_100g"] = X["price_per_100g"].apply(np.log1p)
y = df["rating"]
```

### Pre-processing

Before we can train a model, we need to perform some additional steps. We msut first split the dataset into train/validation/test, where I have chosen a 60%/20%/20% distribution. I have set the `random_state` parameter to 1 to guarantee reproducibility.

```python
from sklearn.model_selection import train_test_split

X_train_val, X_test = train_test_split(X, test_size=0.2, random_state=1)
y_train_val, y_test = train_test_split(y, test_size=0.2, random_state=1)

X_train, X_val = train_test_split(X_train_val, test_size=0.25, random_state=1)
y_train, y_val = train_test_split(y_train_val, test_size=0.25, random_state=1)
```

Lastly we will encode the categorical features using `DictVectorizer`.

```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
dv.fit(X_train.to_dict(orient="records"))


def _transform(df: pd.DataFrame):
    return dv.transform(df.to_dict(orient="records"))
```

### Linear regression

Let's start with the simplest model which is a linear regressor. I have used the `Ridge` model which included L2 regularisation to prevent overfitting. I will train the model for multiple values of the regulurisation weight parameter `alpha`, and recorded the losses on the training and validation sets.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

scores_linear = pd.DataFrame(columns=["train", "validation"])
for alpha in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(_transform(X_train), y_train)
    scores.loc[alpha, :] = pd.Series(
        {
            "train": mean_squared_error(y_train, model.predict(_transform(X_train)), squared=False),
            "validation": mean_squared_error(y_val, model.predict(_transform(X_val)), squared=False),
        }
    )

fig = scores_linear.plot(log_x=True, labels={"index":"alpha", "value":"loss"})
```

{{< include src="charts/linear_losses.html" >}}

We see that if we use too high a value of `alpha`, the RMSE starts to increase because the regularisation term is too strong and forces too simple a model. It seems that a suitable value of `alpha` is 1.0, since this gives a relatively low loss on both the validation and test sets.

We can now fit a model on the combined train and validation set.

```python
linear_model = Ridge(alpha=1.0)
linear_model.fit(_transform(X_train_val), y_train_val)
```

If we plot the predicted distribution of ratings from this model, we see that it captures the central part of the distribution quite well. However, the model fails to predict the more extreme ratings, and therefore the peak around the median is slightly higher.

```python
pd.DataFrame(
    {"true": y_train_val, "prediction": np.round(linear_model.predict(_transform(X_train_val)), decimals=0)}
).hist(histnorm="percent", barmode="group", labels={"value": "rating [%]"})
```

{{< include src="charts/linear_hist.html" >}}

We can get a bit more insight by evaluating the importance of the difference features using `permutation_importance`.

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(linear_model, _transform(X_train_val), y_train_val, n_repeats=10, random_state=0)
linear_importances = pd.Series(dict(zip(dv.get_feature_names_out(), r.importances_mean)))
linear_importances[linear_importances.abs().sort_values(ascending=False).index]
```

| feature                        | importance |
| ------------------------------ | ---------- |
| price_per_100g                 | 0.243042   |
| roaster=Other                  | 0.075393   |
| flavours=resinous              | 0.066831   |
| region_of_origin=East Africa   | 0.063009   |
| roaster_country=Taiwan         | 0.059065   |
| roast=Medium-Light             | 0.053463   |
| roaster_country=United States  | 0.042189   |
| flavours=fruity                | 0.038182   |
| roaster=Hula Daddy Kona Coffee | 0.029996   |
| roaster=Kakalove Cafe          | 0.029231   |
| roast=Light                    | 0.026638   |

We can see in both cases that the biggest influence is the price. This suggests that either:

- Price is genuinely an indicator of quality
- Price biases the reviewers

Other than the price, the region of origin plays a big influence. Whether a coffee is roasted by certain roasters also has a relatively significant impact on the rating. Surprisingly the flavour notes do not have that much influence apart from the "resinous" and "fruity" flavours.

### Gradient-boosted trees

Another type of model which performs well on tasks like this are random forests. Here I will use `XGBoost` which also performs [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to further improve performance. We will train a model for an increasing number of estimators (ie decision trees), and for increasing maximum tree depth (set by the `max_depth` parameter).

```python
import xgboost as xgb

eval_sets = {
    "train": (_transform(X_train), y_train),
    "validation": (_transform(X_val), y_val),
}

scores_depth = {}
for max_depth in [1, 2, 3, 4, 5]:
    xgb_params = {
        'max_depth': max_depth,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.XGBRegressor(**xgb_params, eval_metric="rmse")
    model.fit(_transform(X_train), y_train, eval_set=list(eval_sets.values()))

    results = model.evals_result()
    scores[max_depth] = pd.DataFrame({k: results[f"validation_{i}"]["rmse"] for i, k in enumerate(eval_sets)})

import plotly.graph_objects as go
COLORS = px.colors.qualitative.Plotly
fig = go.Figure()
for i, (depth, df) in enumerate(scores_max_depth.items()):
    fig.add_trace(go.Scatter(x = df.index, y=df["train"], name=f"{depth} (train)", line_dash="dash", line_color=COLORS[i]))
    fig.add_trace(go.Scatter(x = df.index, y=df["validation"], name=f"{depth} (val)", line_color=COLORS[i]))
fig.update_layout(xaxis_title="n_estimators", yaxis_title="rmse", legend_title_text = "max_depth")
fig.show()
```

{{< include src="charts/trees_losses_depth.html" >}}

We can see that the model is able to achieve a lower RMSE for greater values for `max_depth`. However, this does not come with a corresponding decrease in validation error, indicating that there is overfitting. A suitable value would be `max_depth`=2 or 3 since this is the lowest depth which achieves the minimum validation loss of ~1.05. Let us select `max_depth`=3 with 10 estimators, since this gives the lowest validation loss without any indication of overfitting at low numbers of estimators.

We can now retrain a model with these parameters, but for multiple values of the `eta` parameter which controls the "learning rate" (ie how strongly each new estimator aims to compensate for the previous ones):

```python
scores_eta = {}
for eta in [0.01, 0.03, 0.1, 0.3, 1.0]:
    xgb_params = {
        'max_depth': 3,
        'n_estimators': 10,
        "eta": eta,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.XGBRegressor(**xgb_params, eval_metric="rmse")
    model.fit(_transform(X_train), y_train, eval_set=list(eval_sets.values()))

    results = model.evals_result()
    scores_eta[eta] = pd.DataFrame({k: results[f"validation_{i}"]["rmse"] for i, k in enumerate(eval_sets)})

fig = go.Figure()
for i, (eta, df) in enumerate(scores_eta.items()):
    fig.add_trace(go.Scatter(x = df.index, y=df["train"], name=f"{eta} (train)", line_dash="dash", line_color=COLORS[i]))
    fig.add_trace(go.Scatter(x = df.index, y=df["validation"], name=f"{eta} (val)", line_color=COLORS[i]))

fig.update_layout(xaxis_title="n_estimators", yaxis_title="rmse", legend_title_text = "eta")
```

{{< include src="charts/trees_losses_eta.html" >}}

We can see that if we set too low a value, the model is not able to achieve as low a loss. However, if it is too high, the model overfits and the training loss continues to decrease without reducing the validation loss. It appears that we should select `eta` = 0.3 to give the best compromise.

We can now train a model with these final parameters on the combined training and validation sets:

```python
xgb_params = {
    'max_depth': 3,
    'n_estimators': 10,
    "eta": 0.3,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'seed': 1,
    'verbosity': 1,
}
xgb_model = xgb.XGBRegressor(**xgb_params, eval_metric="rmse")
xgb_model.fit(_transform(X_train_val), y_train_val, eval_set=[(_transform(X_train_val), y_train_val)])
results = xgb_model.evals_result()
```

In the same way as the linear model, this model fails to capture the very low or high ratings.

```python
pd.DataFrame(
    {
        "true": y_train_val,
        "prediction": np.round(xgb_model.predict(_transform(X_train_val)), decimals=0),
    }
).hist(histnorm="percent", barmode="group", labels={"value": "rating [%]"})
```

{{< include src="charts/trees_hist.html" >}}

As with the linear models, can get a bit more insight by evaluating the importance of the difference features.

```python
r = permutation_importance(xgb_model, _transform(X_train_val), y_train_val, n_repeats=10, random_state=0)
xgb_importances = pd.Series(dict(zip(dv.get_feature_names_out(), r.importances_mean)))
xgb_importances[xgb_importances.abs().sort_values(ascending=False).index]
```

| feature                        | importance |
| ------------------------------ | ---------- |
| price_per_100g                 | 0.306629   |
| flavours=resinous              | 0.093346   |
| roaster=Other                  | 0.062972   |
| roaster_country=Taiwan         | 0.060680   |
| flavours=fruity                | 0.040122   |
| region_of_origin=North America | 0.030498   |
| region_of_origin=East Africa   | 0.027630   |
| roast=Medium-Light             | 0.018537   |
| roaster=El Gran Cafe           | 0.016596   |
| roast=Medium-Dark              | 0.015745   |
| roaster=Kakalove Cafe          | 0.010607   |
| flavours=nutty                 | 0.010169   |

We see that the price continues to have the biggest influence. We also see that the "resinous" and "fruity" flavours are significant here. The roaster features also play a significant role, but interestingly the roasters which feature here are different from the linear model.

### Model selection

Finally can evaluate the losses of both models on the test dataset, and see which model is most accurate.

```python
models = {"linear": linear_model, "xgb": xgb_model}

scores_comparison = pd.DataFrame(dtype=float)
for name, model in models.items():
    loss_train = mean_squared_error(y_train_val, model.predict(_transform(X_train_val)), squared=False)
    loss_test = mean_squared_error(y_test, model.predict(_transform(X_test)), squared=False)
    scores[name] = pd.Series({"train": loss_train, "test": loss_test})

fig = px.bar(scores_comparison.transpose(), barmode="group", labels={"index": "model", "value": "loss"})
```

{{< include src="charts/comparison_losses.html" >}}

Both models achieve similar losses on the test set. However, we can also see that the `XGBoost` model has a much larger difference between the train and test losses. This suggests that this model is too simple and has overfit to the training set.

We can see that the two models predict a similar distribution of ratings with shorter tails that the ground truth distribution.

```python
pd.DataFrame(
    {"true": y_test} | {name: np.round(model.predict(_transform(X_test)), decimals=0) for name, model in models.items()}
).hist(histnorm="percent", barmode="group")
```

{{< include src="charts/comparison_hist.html" >}}

This suggests that the model is not the reason for failing to predict the highest/lowest scores is more due to some other more systematic error such as:

- Lack of information in the features (eg perhaps we need more detailed information about the origin)
- System error in the reviews (eg different reviewers)

## Conclusions

We have shown that it is possible to predict how highly rated a coffee would be on CoffeeReview.com based purelty on information about the coffee and a linear model. We have shown that the biggest influencers on rating are:

- Price
- Flavour: if a coffee is fruity or resinous
- If a coffee is from East Africa
- If a coffee is from certain roasters

We showed that the models could predict the rating to quite a high degree of accuracy, but struggled to predict particularly low or highly rated coffees. However, they produced an unbiased estimate.

Overall, the two models have very similar performance on the test set. The linear regression model is preferred since it is simpler and has _slightly_ better performance.

In the [next post]({{< ref "../deployment" >}}) in this series, we will actually deploy the trained model to the cloud.
