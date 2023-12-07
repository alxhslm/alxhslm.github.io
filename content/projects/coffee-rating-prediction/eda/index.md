+++
title = 'Coffee Rating Prediction: EDA'
series = ['Coffee Rating Prediction']
series_order = 1
date = 2023-11-09T00:00:00Z
showTableOfContents = true
+++

This was my first personal ML project. The goal was to familiarise myself with [XGBoost](https://xgboost.readthedocs.io/en/stable/) and [AWS Lambda](https://aws.amazon.com/lambda/). I enjoy drinking nice coffee, so I chose a topic which I hoped would help me buy better coffee in the future. The source code is available on GitHub.

{{< github repo="alxhslm/coffee-rating-prediction" >}}

## Objective

The objective of this project is to be able to predict how highly rated a coffee would be on [CoffeeReview.com](http://CoffeeReview.com) based purely on information about the coffee such as:

- Origin
- Roaster and roasting style
- Price
- Flavour profile

I will use this [dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset/data) from Kaggle which contains ratings for ~1900 coffees.

## Data processing

I will begin by loading in the data into a `pd.DataFrame` and also renaming the columns to more clearly distinguish between the country of origin and roaster country.

```python
import pandas as pd

df = pd.read_csv("./data/simplified_coffee.csv")
for col in ["name", "roaster", "roast", "loc_country", "origin", "review"]:
    df[col] = df[col].astype("string")

df["review_date"] = pd.to_datetime(df["review_date"])
df = df.rename(columns={"loc_country": "roaster_country", "100g_$": "price_per_100g", "origin": "country_of_origin"})
df.head()
```

Let us first check for NaNs.

```python
df.isna().sum()
```

| name              | 0   |
| ----------------- | --- |
| roaster           | 0   |
| roast             | 12  |
| roaster_country   | 0   |
| country_of_origin | 0   |
| price_per_100g    | 0   |
| rating            | 0   |
| review_date       | 0   |
| review            | 0   |

The only column with NaNs is the roast. Since there are only 12 missing values, we could just remove these rows. However, since most coffees have the same roast type (as will see later), let us fill with the modal value.

```python
df["roast"] = df["roast"].fillna(df["roast"].mode().iloc[0])
```

Let's fix a typo in the roaster country for one coffee.

```python
df["roaster_country"] = df["roaster_country"].str.replace("New Taiwan", "Taiwan")
```

Some roasters such as "El Gran Cafe" were entered with slightly different spellings (eg sometimes with the accented é and sometimes with a plain e). I will rename the roasters consistently by replacing these characters:

```python
replace = {"’s": "'s", "é": "e", "’": "'"}
for k, v in replace.items():
    df["roaster"] = df["roaster"].str.replace(k, v)
```

## Distributions of the different columns

### Ratings

We can see that the ratings appear to be approximately normally distributed. However, the median rating is surprisingly high at ~94%.

```python
df["rating"].hist(histnorm='percent', labels={"value":"rating [%]"})
```

{{< include src="charts/ratings_hist.html" >}}

### Price

The distribution for the price of the coffee is shown below.

```python
df["price_per_100g"].hist((histnorm='percent', labels={"value":"price per 100g [$]"})
```

{{< include src="charts/price_hist.html" >}}

There is a very long tail, due to a few very expensive coffees. This suggests that there may be benefit in applying the log transformation. Once we do this, the distribution is closer to a normal distribution.

```python
df["price_per_100g"].apply(np.log1p).(histnorm='percent', labels={"value":"price per 100g [log $]"})
```

{{< include src="charts/price_log_hist.html" >}}

### Roasting style

The vast majority of the coffee have the medium-light roast type. This large bias in the dataset may make it challenging for a model to detect any impact of roast style on coffee rating.

```python
df["roast"].hist(histnorm="percent", labels={"value":"roast"})
```

{{< include src="charts/roast_hist.html" >}}

### Roaster country

Most of the data we have is from US rosters.

```python
df["roaster_country"].value_counts()
```

| roaster_country |     |
| --------------- | --- |
| United States   | 774 |
| Taiwan          | 339 |
| Hawai'i         | 77  |
| Guatemala       | 24  |
| Hong Kong       | 9   |
| Japan           | 8   |
| England         | 7   |
| Canada          | 5   |
| Australia       | 1   |
| China           | 1   |
| Kenya           | 1   |

If we look at the distribution of pricing for the most common countries, we see that the distribution is quite different in each country. In particular, the distribution of coffees from US roasters has a much more pronounced peak at the lower price level. This likely indicates that there is some bias in the dataset. Given that CoffeeReview is based in the US, they have reviewed a disproportionate number of more affordable coffees from US roasters.

```python
import plotly.express as px

countries = ["United States", "Taiwan", "Guatemala"]
px.histogram(
    df[df["roaster_country"].apply(lambda c: c in countries)],
    x="price_per_100g",
    color="roaster_country",
    barmode="group",
    histnorm='percent',
)
```

{{< include src="charts/roaster_country_hist.html" >}}

{{< alert >}}
This strong bias towards coffees roasted in the US means that it is unclear how well our will apply to coffees roasted outside the US. In addition, the number of different countries present is very small, and we cannot for example, predict if a coffee from a German roaster would be more or less likely to be highly rated since there are no coffees from German roasters in the dataset.
{{< /alert >}}

### Country of origin

We will now examine the country of origin for the different coffees:

```python
df["region_of_origin"].hist(histnorm="percent", labels={"value":"country of origin"})
```

{{< include src="charts/origin_country_hist.html" >}}

As expected, most of the coffees come from the largest coffee producing regions in the world. Almost all examples are from one of the following regions:

- East Africa such as Ethiopia or Kenya
- Central or South America such as Colombia or Guatemala

There are also a lot of coffees from Hawaii, which is likely again because the data is from a US-based website.

## Correlations between features

### Roaster

If we look at the highest and lowest rates coffees, we see that they are dominated by certain roasters.

```python
df.loc[df["rating"] > 96, ["name", "roaster"]].groupby("roaster").count()
```

| roaster                    | count |
| -------------------------- | ----- |
| Barrington Coffee Roasting | 1     |
| Bird Rock Coffee Roasters  | 1     |
| Dragonfly Coffee Roasters  | 1     |
| Hula Daddy Kona Coffee     | 1     |
| JBC Coffee Roasters        | 3     |
| Kakalove Cafe              | 1     |
| Paradise Roasters          | 2     |

```python
df.loc[df["rating"] < 90, ["name", "roaster"]].groupby("roaster").count()
```

| roaster      | count |
| ------------ | ----- |
| El Gran Cafe | 8     |
| Other        | 4     |

This suggests that either:

- Certain roasters find the best/worst coffees or roast them particularly well
- The reviewers favour/dislike certainer roasters

In either case, our model may need to access the roaster.

## Conclusions

We have established that:

1. There is a strong bias towards:
   - Coffees from US roasters
   - Coffees from East Africa or South America
   - Medium-light roasted coffees
2. The roaster appears to have a significant impact on the rating of the coffee, but we cannot determine if this is due to the quality of the roasting or reviewer bias.

In the [next post]({{< ref "../model" >}}), we will move onto training a predictive model on the data.
