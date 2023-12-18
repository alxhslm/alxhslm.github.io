+++
title = 'Modelling'
series = ['Coffee Rating Prediction']
series_order = 2
date = 2023-11-13T00:00:00Z
showTableOfContents = true
+++

## Background

In a [previous post]({{< ref "./eda" >}}) in this series, we performed experimental data analysis to understand the distribution of the rating and get some insight as to which features might be important.

The next step is to use this information to develop and train a model on the data which is capable of predicting the ratings.

## Feature selection

We have quite a few features available, but not all will lead to a significant improvement of the accuracy of the model. We will therefore begin by assessing the importance of the different features in a quantitative way, so that we include the minimum number needed to generate accurate predictions.

### Correlation

We can first compute the correlation coefficient between the numerical features and the rating. In this case, the only numerical feature is the `"price_per_100g"`:

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

We can see that the roaster has by far the biggest influence, which was expected based on the results from the EDA. This is followed by the country of origin. Interestingly, the _region_ of origin has a much lower mutual information score than the _country_ of origin, suggesting that we cannot use the region in place of the country of origin.

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

We can see that the most important flavours are the "resinous" and "fruity" flavours, and have a similar level of significance to the `"roast"` feature, which again agrees with the results from the EDA. The "spicy" and "nutty" together provide about the same information as the "fruity" flavour. The rest of the flavours are much less significant.

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
roaster_map = df[["roaster", "roaster_country"]].groupby("roaster").first()["roaster_country"]
roaster_info = {"known_roasters": roaster_map.to_dict(), "popular_roasters": popular_roasters}
with open("data/roasters.json", "w") as f:
    json.dump(roaster_info, f, indent=4)
```

We can now use this information to engineer roaster features with a smaller number of unique values.

```python
df["popular_roaster"] = df["roaster"].where(df["roaster"].apply(lambda r: r in popular_roasters), "Other")
```

{{< alert >}}
If we find these features have a strong influence on the model, we need to be careful when applying the model to new coffees from unknown roasters. Even if the coffee is from a well-known roaster, they will have a `"roaster"` value of "Other" if they are not present in the training set.
{{< /alert >}}

### Flavours

Let's now combine the flavours into a single column.

```python
df["flavours"] = df.apply(lambda coffee: [flavour for flavour in FLAVOURS if coffee[flavour]], axis=1)
```

### One-hot encoding

The remaining task is to encode the categorical variables. We will use a one-hot encoding scheme, which can be easily implemented using the `DictVectorizer`:

```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
dv.fit(X_train.to_dict(orient="records"))
```

Note that this can natively handle the `"flavours"` column which contains a list of flavours. Every time we train or evaluate a model we will need to apply this transformation.

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

X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)
```

We will train the model using the train sets, and finally evaluate using the test set. We will use K-folds validation to guard against overfitting to the validation set when performing hyperparameter selection.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1)
```

### Linear regression

Let's start with the simplest model which is a linear regressor. I will use the `Ridge` model which included L2 regularisation to prevent overfitting. I will train the model for multiple values of the regularisation weight parameter `alpha`, and record the losses on the training and validation sets.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def train_ridge_using_kfold(model: Ridge, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    mse_train = []
    mse_val = []
    for _, (train_index, val_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index, :]
        y_train = y.iloc[train_index]
        X_val = X.iloc[val_index, :]
        y_val = y.iloc[val_index]
        model.fit(_transform(X_train), y_train)
        mse_train.append(mean_squared_error(y_train, model.predict(_transform(X_train))))
        mse_val.append(mean_squared_error(y_val, model.predict(_transform(X_val))))
    return np.mean(mse_train), np.mean(mse_val)

scores_linear = pd.DataFrame(columns=["train", "validation"])
for alpha in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    model = Ridge(alpha=alpha)
    mse_train, mse_val = train_ridge_using_kfold(model, X_train, y_train)
    scores.loc[alpha, :] = pd.Series({"train": mse_train, "validation": mse_val})

fig = scores_linear.plot(log_x=True, labels={"index":"alpha", "value":"loss"})
```

{{< include src="charts/linear_losses.html" >}}

We see that if we use too high a value of `alpha`, the RMSE starts to increase because the regularisation term is too strong and forces too simple a model. It seems that a suitable value of `alpha` is 10.0, since this gives a relatively low loss on both the validation and test sets.

We can now fit a model on the combined train and validation set.

```python
linear_model = Ridge(alpha=10.0)
linear_model.fit(X_train, y_train)
```

If we plot the predicted distribution of ratings from this model, we see that it captures the central part of the distribution quite well and there is no bias. However, the model fails to predict the more extreme ratings, and therefore the peak around the median is slightly higher.

```python
pd.DataFrame(
    {"true": y_train, "prediction": np.round(linear_model.predict(X_train), decimals=0)}
).hist()
```

{{< include src="charts/linear_hist.html" >}}

We can get a bit more insight by evaluating the importance of the different features using `permutation_importance`.

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(linear_model, X_train, y_train, n_repeats=10, random_state=0)
linear_importances = pd.Series(dict(zip(dv.get_feature_names_out(), r.importances_mean)))
linear_importances[linear_importances.abs().sort_values(ascending=False).index]
```

| Feature                                | Importance |
| -------------------------------------- | ---------- |
| price_per_100g                         | 0.177400   |
| popular_roaster=Other                  | 0.078086   |
| flavours=resinous                      | 0.070506   |
| country_of_origin=Kenya                | 0.065933   |
| country_of_origin=Ethiopia             | 0.044702   |
| roaster_country=Taiwan                 | 0.037096   |
| flavours=fruity                        | 0.033723   |
| popular_roaster=Kakalove Cafe          | 0.024013   |
| popular_roaster=Hula Daddy Kona Coffee | 0.018847   |
| country_of_origin=Panama               | 0.017545   |

We can see that the biggest influence is the price, which was expected given the high correlation we observed. Certain countries styles have a large importance, which is expected given that the mutual information from this feature was quite high. The flavours have a significant but lower importance compared to the other features.

### Gradient-boosted trees

Another type of model which performs well on tasks like this are random forests. Here I will use `XGBoost` which also performs [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to further improve performance. We will train a model for an increasing number of estimators (ie decision trees), and for increasing maximum tree depth (set by the `max_depth` parameter).

```python
import xgboost as xgb

def train_xgb_using_k_fold(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    mse = []
    for _, (train_index, val_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index, :]
        y_train = y.iloc[train_index]
        X_val = X.iloc[val_index, :]
        y_val = y.iloc[val_index]
        eval_sets = {
            "train": (_transform(X_train), y_train),
            "validation": (_transform(X_val), y_val),
        }
        model.fit(_transform(X_train), y_train, eval_set=list(eval_sets.values()))
        results = model.evals_result()
        mse.append(pd.DataFrame({k: results[f"validation_{i}"]["rmse"] for i, k in enumerate(eval_sets)}))
    return sum(mse) / len(mse)

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
    scores[max_depth] =train_xgb_using_k_fold(model, X_train, y_train)

import plotly.graph_objects as go
fig = go.Figure()
for i, (depth, df) in enumerate(scores_max_depth.items()):
    fig.add_trace(go.Scatter(x = df.index, y=df["train"], name=f"{depth} (train)", line_dash="dash", line_color=COLORS[i]))
    fig.add_trace(go.Scatter(x = df.index, y=df["validation"], name=f"{depth} (val)", line_color=COLORS[i]))
fig.update_layout(xaxis_title="n_estimators", yaxis_title="rmse", legend_title_text = "max_depth")
fig.show()
```

{{< include src="charts/trees_losses_depth.html" >}}

We can see that the model is able to achieve a lower RMSE for greater values for `max_depth`. However, this does not come with a corresponding decrease in validation error, indicating that there is overfitting. A suitable value would be `max_depth`=1 or 2 since these have the lowest difference between the training and validation loss. Using a `max_depth` of 2 does lead to a lower validation loss, so this is preferable, so long as the number of estimators is limited. A suitable selection would `max_depth`=2 with 10 estimators.

We can now retrain a model with these parameters, but for multiple values of the `eta` parameter which controls the "learning rate" (ie how strongly each new estimator aims to compensate for the previous ones):

```python
scores_eta = {}
for eta in [0.01, 0.03, 0.1, 0.3, 1.0]:
    xgb_params = {
        'max_depth': 2,
        'n_estimators': 10,
        "eta": eta,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.XGBRegressor(**xgb_params, eval_metric="rmse")
    scores_eta[eta] = train_xgb_using_k_fold(X_train, y_train)

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
    'n_estimators': 10,
    "eta": 0.3,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'seed': 1,
    'verbosity': 1,
}
xgb_model = xgb.XGBRegressor(**xgb_params, eval_metric="rmse")
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train)])
results = xgb_model.evals_result()
```

If we plot the histogram of the model predictions, we see that this model also fails to capture the very low or high ratings in the same way as the linear model.

```python
pd.DataFrame(
    {
        "true": y_train,
        "prediction": np.round(xgb_model.predict(X_train), decimals=0),
    }
).hist()
```

{{< include src="charts/trees_hist.html" >}}

As with the linear models, can get a bit more insight by evaluating the importance of the difference features.

```python
r = permutation_importance(xgb_model, X_train, y_train, n_repeats=10, random_state=0)
xgb_importances = pd.Series(dict(zip(dv.get_feature_names_out(), r.importances_mean)))
xgb_importances[xgb_importances.abs().sort_values(ascending=False).index]
```

| feature                       | importance |
| ----------------------------- | ---------- |
| price_per_100g                | 0.172254   |
| flavours=resinous             | 0.090398   |
| popular_roaster=Other         | 0.069107   |
| flavours=fruity               | 0.038945   |
| popular_roaster=El Gran Cafe  | 0.031304   |
| roaster_country=Taiwan        | 0.024728   |
| country_of_origin=Kenya       | 0.023421   |
| country_of_origin=Ethiopia    | 0.020484   |
| popular_roaster=Kakalove Cafe | 0.013997   |
| country_of_origin=Panama      | 0.010082   |

We see that the price continues to have the biggest influence. We also see that the "resinous" and "fruity" flavours continue to be significant here, and are actually more important than certain countries of origin. The roaster feature also play a significant role, but interestingly the roaster which features here is different from the linear model.

### Model selection

Finally can evaluate the losses of both models on the test dataset, and see which model is most accurate.

```python
models = {"linear": linear_model, "xgb": xgb_model}

scores_comparison = pd.DataFrame(dtype=float)
for name, model in models.items():
    loss_train = mean_squared_error(y_train, model.predict(X_train), squared=False)
    loss_test = mean_squared_error(y_test, model.predict(X_test), squared=False)
    scores[name] = pd.Series({"train": loss_train, "test": loss_test})

scores_comparison.transpose().plot.bar()
```

{{< include src="charts/comparison_losses.html" >}}

Both models achieve similar losses on the training set, but there is difference in performance on the test set. We can see that the `XGBoost` model has a much larger difference between test losses. This suggests that this model has overfit to the training set.

We can see that the two models both predict a similar distribution of ratings with shorter tails than the ground truth distribution.

```python
pd.DataFrame(
    {"true": y_test} | {name: np.round(model.predict(X_test), decimals=0) for name, model in models.items()}
).hist()
```

{{< include src="charts/comparison_hist.html" >}}

This suggests that the models are failing to predict the highest/lowest scores due to some systematic error such as:

- Lack of information in the features (eg perhaps we need more detailed information about the origin)
- Inconsistencies in the review process (eg different reviewers with different preferences)

Investing this is beyond the scope of this project, since the linear model achieves the objective with sufficiently good performance.

## Conclusions

We have shown that it is possible to predict how highly rated a coffee would be on CoffeeReview.com based purely on information about the coffee and a linear model. We have shown that the biggest influencers on rating are:

- Price
- Flavour: if a coffee is fruity or resinous
- Origin: If a coffee is from East Africa
- If a coffee is from certain roasters

We showed that the models could predict the rating to quite a high degree of accuracy, but struggled to predict particularly low or highly rated coffees. However, it produces an unbiased estimate.

Overall, the two models have very similar performance on the test set. The linear regression model is preferred since it is simpler and has _slightly_ better performance.

In the [next post]({{< ref "./deployment" >}}) in this series, we will actually deploy the trained model to the cloud.
