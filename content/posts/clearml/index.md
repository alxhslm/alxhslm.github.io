+++
title = 'ClearML: A review'
date = 2023-12-30T14:04:46Z
tags = ['clearml', 'machine learning']
+++

When building ML applications, managing the various datasets & models can be quite a headache, which is where MLOps tools come in. The current landscape is still quite immature, and there are endless tools with different capabilities, aiming to streamline different aspects of the machine learning lifecycle. This makes is quite difficult to assess which tools are suitable for your use case.

In this blog post, I will share my experiences using ClearML, which is one of the most popular MLOps tools. I will discuss the features which we found to be useful at my current company, as well as some of the limitations we ran into. I hope this will help guide others when choosing an MLOps tool for their team.

## What is ClearML?

[ClearML](http://clear.ml/) is a popular end-to-end MLOps tool that covers the whole ML pipeline within a single open-source platform. It offers suite of tools to manage datasets, experiments, and models, and also track them all in a unified web interface. There are some other tools in this space, and I have a put a brief comparison of the other popular ones below. Compared to the other options, ClearML is a more heavyweight tool, although not quite as popular.

|                     | [ClearML](http://clear.ml/) | [MLflow](https://mlflow.org/docs/latest/index.html) | [Weights and Biases](https://wandb.ai/) |
| ------------------- | --------------------------- | --------------------------------------------------- | --------------------------------------- |
| Open-source         | :white_check_mark:          | :white_check_mark:                                  | :white_check_mark:                      |
| Self-host           | :white_check_mark:          | :white_check_mark:                                  | :white_check_mark:                      |
| Cloud-hosted        | :white_check_mark:          | :x:                                                 | :white_check_mark:                      |
| Dataset management  | :white_check_mark:          | :x:                                                 | :x:                                     |
| Experiment tracking | :white_check_mark:          | :white_check_mark:                                  | :white_check_mark:                      |
| Model registry      | :white_check_mark:          | :white_check_mark:                                  | :x:                                     |
| Model deployment    | :white_check_mark:          | :white_check_mark:                                  | :x:                                     |

[![Star History Chart](https://api.star-history.com/svg?repos=allegroai/clearml,mlflow/mlflow,wandb/wandb&type=Date)](https://star-history.com/#allegroai/clearml&mlflow/mlflow&wandb/wandb&Date)

## Why did we choose ClearML?

### Flexible experiment tracking

One of the major strengths of ClearML is that it is quite generic, making it suitable for a wide range of ML applications. ClearML allows users to record custom metadata, log live metrics and store plots and other rich data in the results.

For example, at my company, we used ClearML for the classical use-case of training models implemented in JAX, where we tracked hyperparameters, and recorded the trained parameters and other metrics. However, we also needed to run computationally intensive simulations with these models. In this case, we used ClearML to record settings for the simulations, and track the progress so we could monitor the results in real-time. ClearML was able to support both use-cases very well.

### Run experiments anywhere

ClearML experiments can run on any machine where you can run the agent. We initially used ClearML to track experiments run locally on our machines. Once the number of models we needed to maintain increased, we then built the capability to run experiments in the cloud.

The value of tracking _local_ experiments is often underestimated during the model development process, but it allowed us to collaborate more easily and isolate performance regressions. Being able to train models in the cloud then allowed us to iterate more quickly, but tracking experiments locally was still a big step forward [^1].

[^1]: It is worth noting that this also applies to other MLOps tools in general, and not just ClearML.

### Vibrant community

ClearML offers comprehensive and well-structured documentation, particularly for the most-used APIs. It was easy for us to find the information we needed to get started and make the most out of the platform.

Additionally, ClearML has an active community on [Slack](https://clear.ml/community), providing a platform for users to seek support, share ideas, and collaborate with other users. Whenever we ran into issues or found bugs, we found that the ClearML team and wider community was quick to respond.

### Batteries included

Unlike some other tools such as Weights and Biases, ClearML goes beyond just experiment tracking, and includes features for [dataset management](https://clear.ml/docs/latest/docs/clearml_data/), [orchestration](https://clear.ml/docs/latest/docs/pipelines/) and even [deployment](https://clear.ml/docs/latest/docs/clearml_serving/). This allows you have a centralised platform for the whole ML pipeline, without having to integrate additional tools such as using [DVC](https://dvc.org/) to version datasets or [KServe](https://kserve.github.io/website/latest/) for deployment, which could be seen as introducing additional complexity.

Whether this is seen as positive or negative will likely depend on your use case. However, if you need the full set of capabilities offered by ClearML and have a small team, it is much quicker to get up and running with a single tool.

### Affordable pricing for small teams

One of the main reasons we chose ClearML was that it offers a [Community Edition](https://github.com/allegroai/clearml/) which allows users to self-host on their own infrastructure. This allows us to keep all of the data in-house and reduce the running costs significantly, which is obviously a significant advantage for a start-up.

ClearML also offers a hosted solution if you don’t want to manage your own infrastructure. This is priced based on the number of users, which means it is affordable for small teams, although this might become expensive once your team scales.

## What limitations have we found with ClearML?

### Some key features are paywalled

The free tier and self-hosted options are generally quite generous. However, not _all_ features you might expect included, and the [documentation](https://clear.ml/pricing) on which features are included could be made clearer. For example, one limitation we ran into was that autoscalers are not available in the free tier, so we had to build extra tooling in-house to distribute experiments between workers.

Perhaps more disappointingly, other important features such as SSO or Kubernetes integration are not available even in the pro tier, and require one of the enterprise tiers[^2]. Therefore it is important to check if the specific features required for your use case are included in the relevant tier you intend to use.

[^2]: There are both scale and enterprise tiers which both have custom pricing.

### Documentation is patchy

While ClearML’s documentation is generally very clear and quite comprehensive, there are certain more advanced or esoteric areas of the API which are less well-documented. For example, the standard ClearML API does not include support for managing queues (for orchestrating tasks over workers), and instead recommends using the generic [HTTP API client](https://clear.ml/docs/latest/docs/clearml_sdk/apiclient_sdk/). However, the documented for this API is quite limited so some trial and error was required for us to get this to work.

The ClearML documentation also sometimes fails to explain certain concepts beyond the key ones such as `Task`s and `Model`s particularly well. For example, we wanted to run groups of simulations, where each simulation was for a different scenario. The solution turned out to be to use [function tasks](https://clear.ml/docs/latest/docs/references/sdk/task#create_function_task), which allow a parent task to spawn multiple sub-tasks, but it was not clear from the documentation alone that this would achieve what we needed.

Fortunately it seems that ClearML has since improved the documentation in this and other areas, so perhaps this will become less of an issue going forward.

### Visualisation options are limited

ClearML allows you to view metrics from a single experiment very quickly and easily from the UI. You can also generate custom plots and store them in the results for the experiment, which can then also be viewed from the UI.

Whilst you can also very easily compare _metrics_ from different experiments on the same chart, you are unable to view _custom plots_ from different experiments on the same chart [^3]. This means that your visualisation options for comparing experiments is [limited to simple scalar metrics](https://clear.ml/docs/latest/docs/webapp/webapp_exp_comparing).

[^3]: Unfortunately it doesn't seem like this is something which will be supported anytime soon going by this long-standing GitHub [issue](https://github.com/allegroai/clearml/issues/81).

We overcame this limitation by building custom dashboards using Streamlit to fetch data from the ClearML database, and then visualise results using our own custom plots. This worked quite well for us, but this obviously introduces additional maintenance overhead. It would be cleaner if you could analyse experiments in more detail inside the ClearML UI itself.

## Conclusions

With its extensive documentation and active community support, ClearML is an affordable and flexible solution for smaller teams building out ML pipelines. Although ClearML has some shortcomings and certain restrictions in the free/self-hosted tier, it has a much greater set of capabilities compared to the alternatives, and continues to evolve and improve over time. Overall, ClearML has proven to be a valuable MLOps tool at my company, and is an option worth considering for your team.
