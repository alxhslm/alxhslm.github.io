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

## Feature selection

We have quite a few features available, but not all will lead to a significant improvement of the accuracy of the model. We will therefore begin by assessing the importance of the different features in an quantitative way, so that we include the minimum number needed to generate accurate predictions.

### Correlation

We can first compute the correlation coefficient between the numerical features and the rating. In this case, the only numerical feature is the "price_per_100g":

```python
df[["price_per_100g"]].corrwith(df["rating"])
```

| Feature        | Correlation coefficient |
| -------------- | ----------------------- |
| price_per_100g | 0.241615                |

This is a moderately high value, suggesting that the price does significantly influence the rating and should therefore be included as an input to the model.

### Mutual information

For the categorical features, we can instead analyse the [mutual information](https://en.wikipedia.org/wiki/Mutual_information):

```python
from sklearn.metrics import mutual_info_score

mutual_info = pd.Series({k: mutual_info_score(df[k], df["rating"]) for k in ["roaster", "roast", "roaster_country", "country_of_origin", "region_of_origin"]})
mutual_info.sort_values(ascending=False)
```

| Feature           | Mutual information |
| ----------------- | ------------------ |
| roaster           | 0.670450           |
| country_of_origin | 0.159215           |
| roaster_country   | 0.068317           |
| roast             | 0.046098           |
| region_of_origin  | 0.033712           |

We can see that the roaster has by far the biggest influence, which was expected based on the results from the EDA. This is followed by the country of origin. Interestingly, the _region_ of origin has a much lower mutual information score than the _country_ of origin, suggesting that we cannot use this in place of the country of origin.

We can also compute the same metric for the different flavours:

```python
mutual_info_flavours = pd.Series({k: mutual_info_score(df[k], df["rating"]) for k in FLAVOURS})
mutual_info_flavours.sort_values(ascending=False)
```

| Feature   | Mutual information |
| --------- | ------------------ |
| resinous  | 0.046254           |
| fruity    | 0.036157           |
| spicy     | 0.020958           |
| nutty     | 0.012289           |
| acidic    | 0.008529           |
| floral    | 0.006981           |
| chocolate | 0.006747           |
| herby     | 0.006218           |
| carbony   | 0.005304           |
| caramelly | 0.004110           |

We can see that the most important flavours are the "resinous" and "fruity" flavours, and have a similar level of significance to the "roast" feature, which again agrees with the results from the EDA. The "spicy" and "nutty" together provide about the same information as the "fruity" flavour. The rest of the flavours are much less significant.

### Summary

Based on this brief analysis, we will choose the following features:

- Price per 100g
- Roaster
- Roaster country
- Roast
- Country of origin
- Flavours: "resinous", "fruity", "spicy", "nutty"

## Feature engineering

Now that we have selected the features to use, we need to transform them into a form which the model can accept.

### Price

The price of the coffee was found to have a very long tail. Many models make assumptions about the data, including that the features are normally distributed. Therefore we may get better performance if we transform this feature to be closer to a normal distribution. The log transformation is one such transformation which can achieve this:

```python
df["price_per_100g"].apply(np.log1p).hist()
```

{{< include src="charts/price_log_hist.html" >}}

### Roaster

Since the previous analysis showed that there is evidence that the roasters significantly influences the rating, the model needs a feature giving it this information. We cannot simply convert the roaster using one-hot encoding as there are too many different values. Let us instead only include the most common roasters (those with > 10 coffees).

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

We can now use this information to engineer roaster features with a smaller number of unique values.

```python
df["popular_roaster"] = df["roaster"].where(df["roaster"].apply(lambda r: r in popular_roasters), "Other")
```

{{< alert >}}
If we find these features have a strong influence on the model, we need to be careful when applying the model to new coffees from unknown roasters. Even if the coffee is from a well-known roaster, they will have a "roaster" value of "Other" if they are not present in the training set.
{{< /alert >}}

### One-hot encoding

The remaining task is to encode the categorical variables. We will use a one-hot encoding scheme, which can be easily implemented using the `DictVectorizer`:

```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
dv.fit(X_train.to_dict(orient="records"))
```

Note that this can natively handle the "flavours" column which contains a list of flavours. Every time we train or evaluate a model we will need to apply this transformation.

### Summary

We can now combine these steps to assemble the input feature matrix `X` and target vector `y`.

```python
FEATURES = ["price_per_100g", "popular_roaster", "roaster_country", "roast", "country_of_origin"]
FLAVOURS = ["fruity", "resinous", "spicy", "nutty"]
X = df[FEATURES].copy()
X["price_per_100g"] = X["price_per_100g"].apply(np.log1p)
X["flavours"] = df.apply(lambda coffee: [flavour for flavour in FLAVOURS if coffee[flavour]], axis=1)
X = dv.transform(X.to_dict(orient="records"))
y = df["rating"]
```

## Building a model

### Validation framework

We must first split the dataset into train/validation/test sets, where I have chosen a 60%/20%/20% distribution. I have set the `random_state` parameter to 1 to guarantee reproducibility.

```python
from sklearn.model_selection import train_test_split

X_train_val, X_test = train_test_split(X, test_size=0.2, random_state=1)
y_train_val, y_test = train_test_split(y, test_size=0.2, random_state=1)

X_train, X_val = train_test_split(X_train_val, test_size=0.25, random_state=1)
y_train, y_val = train_test_split(y_train_val, test_size=0.25, random_state=1)
```

We will train the model using the train and validation sets, and finally evaluate using the test set.

### Linear regression

Let's start with the simplest model which is a linear regressor. I have used the `Ridge` model which included L2 regularisation to prevent overfitting. I will train the model for multiple values of the regulurisation weight parameter `alpha`, and recorded the losses on the training and validation sets.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

scores_linear = pd.DataFrame(columns=["train", "validation"])
for alpha in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    scores.loc[alpha, :] = pd.Series(
        {
            "train": mean_squared_error(y_train, model.predict(X_train), squared=False),
            "validation": mean_squared_error(y_val, model.predict(X_val), squared=False),
        }
    )

fig = scores_linear.plot(log_x=True, labels={"index":"alpha", "value":"loss"})
```

{{< include src="charts/linear_losses.html" >}}

We see that if we use too high a value of `alpha`, the RMSE starts to increase because the regularisation term is too strong and forces too simple a model. It seems that a suitable value of `alpha` is 10.0, since this gives a relatively low loss on both the validation and test sets.

We can now fit a model on the combined train and validation set.

```python
linear_model = Ridge(alpha=10.0)
linear_model.fit(X_train_val, y_train_val)
```

If we plot the predicted distribution of ratings from this model, we see that it captures the central part of the distribution quite well. However, the model fails to predict the more extreme ratings, and therefore the peak around the median is slightly higher.

```python
pd.DataFrame(
    {"true": y_train_val, "prediction": np.round(linear_model.predict(X_train_val), decimals=0)}
).hist()
```

{{< include src="charts/linear_hist.html" >}}

We can get a bit more insight by evaluating the importance of the difference features using `permutation_importance`.

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(linear_model, X_train_val, y_train_val, n_repeats=10, random_state=0)
linear_importances = pd.Series(dict(zip(dv.get_feature_names_out(), r.importances_mean)))
linear_importances[linear_importances.abs().sort_values(ascending=False).index]
```

| Feature                       | Importance |
| ----------------------------- | ---------- |
| price_per_100g                | 0.179145   |
| country_of_origin=Kenya       | 0.097453   |
| country_of_origin=Ethiopia    | 0.074986   |
| popular_roaster=Other         | 0.072278   |
| roast=Medium-Light            | 0.066722   |
| flavours=resinous             | 0.062963   |
| roaster_country=Taiwan        | 0.057952   |
| roaster_country=United States | 0.042725   |
| roast=Light                   | 0.039749   |
| flavours=fruity               | 0.033624   |

We can see in both cases that the biggest influence is the price, which was expected given the high correlation we observed. Interestingly, certain countries and roast styles have a large importance, even though the mutual information from these feature were found to be lower than some of the others. The flavours have a significant but lower importance compared to these other features.

### Gradient-boosted trees

Another type of model which performs well on tasks like this are random forests. Here I will use `XGBoost` which also performs [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to further improve performance. We will train a model for an increasing number of estimators (ie decision trees), and for increasing maximum tree depth (set by the `max_depth` parameter).

```python
import xgboost as xgb

eval_sets = {
    "train": (X_train, y_train),
    "validation": (X_val, y_val),
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
    model.fit(X_train, y_train, eval_set=list(eval_sets.values()))

    results = model.evals_result()
    scores[max_depth] = pd.DataFrame({k: results[f"validation_{i}"]["rmse"] for i, k in enumerate(eval_sets)})

import plotly.graph_objects as go
fig = go.Figure()
for i, (depth, df) in enumerate(scores_max_depth.items()):
    fig.add_trace(go.Scatter(x = df.index, y=df["train"], name=f"{depth} (train)", line_dash="dash", line_color=COLORS[i]))
    fig.add_trace(go.Scatter(x = df.index, y=df["validation"], name=f"{depth} (val)", line_color=COLORS[i]))
fig.update_layout(xaxis_title="n_estimators", yaxis_title="rmse", legend_title_text = "max_depth")
fig.show()
```

{{< include src="charts/trees_losses_depth.html" >}}

We can see that the model is able to achieve a lower RMSE for greater values for `max_depth`. However, this does not come with a corresponding decrease in validation error, indicating that there is overfitting. A suitable value would be `max_depth`=1 or 2 since these have the lowest difference between the trainign and validation loss. Using a `max_depth` of 2 does lead to a lower validation loss, so this is preferable, so long as the number of estimators is limited. A suitable selection would `max_depth`=2 with 20 estimators.

We can now retrain a model with these parameters, but for multiple values of the `eta` parameter which controls the "learning rate" (ie how strongly each new estimator aims to compensate for the previous ones):

```python
scores_eta = {}
for eta in [0.01, 0.03, 0.1, 0.3, 1.0]:
    xgb_params = {
        'max_depth': 2,
        'n_estimators': 20,
        "eta": eta,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.XGBRegressor(**xgb_params, eval_metric="rmse")
    model.fit(X_train, y_train, eval_set=list(eval_sets.values()))

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
    'max_depth': 2,
    'n_estimators': 20,
    "eta": 0.3,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'seed': 1,
    'verbosity': 1,
}
xgb_model = xgb.XGBRegressor(**xgb_params, eval_metric="rmse")
xgb_model.fit(X_train_val, y_train_val, eval_set=[(X_train_val, y_train_val)])
results = xgb_model.evals_result()
```

In the same way as the linear model, this model fails to capture the very low or high ratings.

```python
pd.DataFrame(
    {
        "true": y_train_val,
        "prediction": np.round(xgb_model.predict(X_train_val), decimals=0),
    }
).hist()
```

{{< include src="charts/trees_hist.html" >}}

As with the linear models, can get a bit more insight by evaluating the importance of the difference features.

```python
r = permutation_importance(xgb_model, X_train_val, y_train_val, n_repeats=10, random_state=0)
xgb_importances = pd.Series(dict(zip(dv.get_feature_names_out(), r.importances_mean)))
xgb_importances[xgb_importances.abs().sort_values(ascending=False).index]
```

| feature                      | importance |
| ---------------------------- | ---------- |
| price_per_100g               | 0.220393   |
| flavours=resinous            | 0.097563   |
| popular_roaster=Other        | 0.087204   |
| roaster_country=Taiwan       | 0.065037   |
| flavours=fruity              | 0.049080   |
| country_of_origin=Kenya      | 0.028768   |
| popular_roaster=El Gran Cafe | 0.020846   |
| roast=Medium-Light           | 0.020247   |
| country_of_origin=Ethiopia   | 0.018645   |
| country_of_origin=Panama     | 0.018514   |

We see that the price continues to have the biggest influence. We also see that the "resinous" and "fruity" flavours continue to be significant here, as well as certain counties of origin. The roaster feature also play a significant role, but interestingly the roaster which features here is different from the linear model.

### Model selection

Finally can evaluate the losses of both models on the test dataset, and see which model is most accurate.

```python
models = {"linear": linear_model, "xgb": xgb_model}

scores_comparison = pd.DataFrame(dtype=float)
for name, model in models.items():
    loss_train = mean_squared_error(y_train_val, model.predict(X_train_val), squared=False)
    loss_test = mean_squared_error(y_test, model.predict(X_test), squared=False)
    scores[name] = pd.Series({"train": loss_train, "test": loss_test})

scores_comparison.transpose().plot.bar()
```

{{< include src="charts/comparison_losses.html" >}}

Both models achieve similar losses on the test set. However, we can also see that the `XGBoost` model has a much larger difference between the train and test losses. This suggests that this model is too simple and has overfit to the training set.

We can see that the two models predict a similar distribution of ratings with shorter tails that the ground truth distribution.

```python
pd.DataFrame(
    {"true": y_test} | {name: np.round(model.predict(X_test), decimals=0) for name, model in models.items()}
).hist()
```

{{< include src="charts/comparison_hist.html" >}}

This suggests that the model is not the reason for failing to predict the highest/lowest scores is more due to some other more systematic error such as:

- Lack of information in the features (eg perhaps we need more detailed information about the origin)
- Systematic error in the review process (eg different reviewers with different preferences)

## Conclusions

We have shown that it is possible to predict how highly rated a coffee would be on CoffeeReview.com based purely on information about the coffee and a linear model. We have shown that the biggest influencers on rating are:

- Price
- Flavour: if a coffee is fruity or resinous
- If a coffee is from East Africa
- If a coffee is from certain roasters

We showed that the models could predict the rating to quite a high degree of accuracy, but struggled to predict particularly low or highly rated coffees. However, they produced an unbiased estimate.

Overall, the two models have very similar performance on the test set. The linear regression model is preferred since it is simpler and has _slightly_ better performance.

In the [next post]({{< ref "../deployment" >}}) in this series, we will actually deploy the trained model to the cloud.
