+++
title = 'Deployment'
series = ['Coffee Rating Prediction']
series_order = 3
date = 2023-11-21T20:12:00Z
showTableOfContents = true
+++

## Background

In a [previous post]({{< ref "./model" >}}) in this series, we trained a model which was able to predict how highly rated a coffee would be on [CoffeeReview.com](http://CoffeeReview.com) based on the following features:

- Origin
- Roaster and roasting style
- Price
- Flavour profile

I will use this [dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset/data) from Kaggle which contains ratings for ~1900 coffees.

## Model deployment

To allow us to interact with the model, it was deployed to the cloud within a server as shown below. Separating the model like this allows it to be updated independently from other systems which use it. The individual components will be discussed further below.

{{< mermaid >}}
graph TD
E[User]-.->A
A[Streamlit Web App]--Request-->B[AWS Lambda]
B--Rating-->A
B--Features-->D[Model]
D--Prediction-->B
subgraph Docker container
D
end
{{< /mermaid >}}

### Docker

The chosen model was wrapped within a Docker container which contains all dependencies including Python, as well as the required packages using [`poetry`](https://python-poetry.org/). The image was built locally and then uploaded to Amazon Container Registry. To update the model, we just need to build a new version of the Docker container.

### AWS Lambda

An [AWS Lambda](https://aws.amazon.com/lambda/) function was then configured to use this Docker container and call the model. This is a serverless product, which means that:

- We don't have to maintain any infrastructure
- We are only billed when the model is actually used

To allow us to interact with the model, an HTTP endpoint was created using a [Lambda URL](https://docs.aws.amazon.com/lambda/latest/dg/lambda-urls.html). In order to secure the endpoint and only allow requests from the dashboard, it was authenticated using IAM. When a HTTP request containing the features is received, the data is passed to the model which generates predictions, and the rating is returned in the response.

## Interactive dashboard

In order to be able to interact the model and predict ratings for arbitrary coffees, I created a [Streamlit](https://streamlit.io/) dashboard which you can access [here](https://coffee-rating-prediction.streamlit.app/). The user enters the meta data about the coffee, and the dashboard then generates a `dict` of the required features.

The dashboard then makes an HTTP request to the server with this `dict` in the message body, and receives the rating in the response. The predicted rating is then displayed in the dashboard. Have a play and see how your favourite coffee fares!

![Streamlit app](../images/streamlit_app.png)
