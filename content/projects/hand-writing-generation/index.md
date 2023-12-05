+++
title = 'Hand Writing Generation'
date = 2023-12-01T21:28:11Z
+++

{{< katex >}}


This was a more advanced project to gain more practical experience with deep learning with unstructured data. With all of the recent hype around generative AI, also thought it would be interesting to use some form of generative model. The source code is available on GitHub.

{{< github repo="alxhslm/hand-writing-generation" >}}


## Objective  
The aim of this project is to: 

1. Train a model to classify hand-written alphanumeric characters
2. Generate synthetic hand-written characters using the same model

I decided to use a [Variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) since it is one of the few widely used models uniting deep learning and Bayesian methods.

## Interactive dashboard
In order to get a better feel for how the model works, and its stengths and weaknesses, I created a Streamlit dashboard (which you can access [here](https://hand-writing-generation.streamlit.app/) to have a go yourself).

![Streamlit app](images/streamlit_app.png)

This dashboard goes through the following steps:

1. You being by drawing a few characters which gives us some observations \\(x_t\\) This allows the encoder to produce an estimate of the posterior distribution \\(p(z|x_{t})\\), which effectively allows the model to “learn” your writing style.
2. You can then generate arbitrary hand-written characters by sampling the distribution \\(p(x) = p(x|z)p(z)\\) where \\(p(z)=p(z|x_{t})\\). 

By varying how you draw, you can change the style of characters which the model produces. Here are some examples of the sort of hand-written characters the model can synthesise:

| Description | Generated images                        |
|-------------|-----------------------------------------|
| Narrow      | ![Narrow characters](images/narrow.jpg) |
| Wide        | ![Wide characters](images/wide.jpg)     |
| Light       | ![Thin characters](images/light.jpg)    |
| Bold        | ![Bold characters ]( images/bold.jpg )  |
| Italic      | ![Italic characters](images/italic.jpg) |
