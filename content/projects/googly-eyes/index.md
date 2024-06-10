+++
title = 'Googly Eyes'
date = 2024-02-08T14:00:00Z
tags = ['machine learning', 'computer vision', 'deployment']
+++

{{< katex >}}

This started as a take-home task for a job application, but I found that the results were quite fun, so I decided to tidy it up and deploy it. You can access the deployed Streamlit dashboard [here](https://googly-eyes.streamlit.app/).

{{< github repo="alxhslm/googly-eyes" >}}

## Task description

FunnyFaces Inc. has hired your talent to develop a web application that allows a user to upload a picture of their face and it returns that same picture but with googly eyes in place of their actual eyes. Fun for the whole family!

The project owner was kind enough to supply a draft of what the company requires; which is attached to this email.

Please develop the "Googly Eyes" service, which exposes a single HTTP endpoint that allows a user to upload a photo and returns the same photo but modified with googly eyes in place of their eyes. Googly eyes are funnier if they are slightly randomised both in size and orientation of the pupils, so that's something you should take into consideration when developing your service. It also shouldn't matter if there are one or more people in the photo - all get googly eyes. As stated - this is fun for the whole family!

Note: FunnyFaces Inc. is a fun company but takes its users' privacy very seriously. The company requires that you **never** store any uploaded picture on your service other than exceptionally for the duration of the request.

![Onfido Senior Machine Learning Engineer Task.jpg](images/Onfido_Senior_Machine_Learning_Engineer_Task.jpg)

## Proposed solution

### Assumptions

To simplify the problem, I made the following assumptions:

- We only wish to detect human faces (and not pets)
- The faces are approximately aligned with the vertical axis of the photo, so the face detection step does need to be robust to misalignment
- The people will be facing the camera in the photos (or close to it), so we do not need to consider 3D geometric effects when placing the eyes

### Architecture

This problem can be decomposed into the following steps:

- Build a method for identifying eyes in photos of people
- Generate Googly eyes with random shapes and sizes
- Create a server which processes HTTP requests
- Create a way to interact with the service

I will now explain how I achieved each of these steps.

## Eye identification

This is an _object localisation_ problem since:

1. We need to identify the locations of eyes in the image
2. We only need to classify a single class of objects

The most obvious approach for identifying eyes is to use a CNN, trained on pre-labelled images of people’s faces.

I considered two different approaches to identifying faces.

### Training my own model

I could think of two possible architectures of identifying the faces:

1. Train a model to detect eyes directly from the photo
2. Use a 2-stage classifier where:
   1. Stage 1: Identify faces
   2. Stage 2: Identify eyes from each face

The latter approach would likely be more robust, since it forces the context around the eyes to be taken in to account. However, it would be slower since two models need to be evaluated.

In either approach, we can make use of pre-trained image classification models for the main trunk of the model:

- They are often trained in datasets with people in so already have some capability to recognise faces and facial features
- The input layers of the network identify “features” which will transfer well to different applications

The dataset we use to train network is very important. We must ensure that it is representative of the sorts of images which the users will upload. Therefore typical datasets for biometric identification would not be suitable. Instead we should use a dataset with photos in more natural environments. The [LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) dataset seems most suitable.

### Pre-trained models

Since the task of recognising faces and facial features is a very common one, we can make use of pre-existing models to perform this step for us. After a brief search, I came across many open-source models, the two most performant models being:

- **MTCNN** ([paper](https://ieeexplore.ieee.org/document/7553523))**:** [PyTorch implementation](https://github.com/timesler/facenet-pytorch) [TensorFlow implementation](https://github.com/ipazc/mtcnn)
- **RetinaFace (**[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf)): \*\*\*\*[PyTorch implementation](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) [TensorFlow implementation](https://github.com/serengil/retinaface)

RetinaFace generally performs _slightly_ better, likely because it was trained with an augmented set of labels including 1k 3D vertices.

I chose to make use of the existing RetinaFace model, in order to allow me to spend more time on the server and dashboard. This model would likely be much more accurate than any model I would train in a short period of time.

I converted the Tensorflow model to use the Tensorflow-lite runtime instead. This required updating the model and pre-processing steps to use fixed image dimensions. This has two main advantages:

- It significantly reduces the size of the docker images
- It reduces execution time

## Googly-eye generation

Once we have detected the position of the eyes, we need to add the googly eyes in the correct location. I used the [`ImageDraw`](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html) module to perform the image manipulation and draw the eyes. However, I still needed to decide how to size the eyes and where to place the pupils.

### Eye size

To ensure that the googly eyes are of an appropriate size, independent of image dimensions or the distance from the face to the camera, I computed the eye-to-eye separation distance for a given face from:

$$
\Delta_{eye2eye}=\sqrt{(x_r-x_l)^2+(y_r-y_l)^2}
$$

where $x_r, y_r$ are the pixel co-ordinates of each eye. I then set the radius of the googly eyes $r_e$ by:

$$
r_e =\gamma\frac{\Delta_{eye2eye}}{2}
$$

where $\gamma$ is a scaling parameter in the range $0<\gamma<1.0$. I set the default value to 0.5.

### Pupil size

The pupil size $r_p$ was set based on the eye size from:

$$
r_p=\lambda r_{e}
$$

where $\lambda$ is a random variable sampled from the following distribution:

$$
\lambda \sim U(\lambda_1, \lambda_2)
$$

where the default values for the parameters $\lambda_1$ and $\lambda_2$ were set to 0.4 and 0.6 respectively.

### Pupil position and orientation

To randomise the position of the pupil, the position was set by the following equation:

$$
\begin{align*}x_p&=x_e+ (r_e-r_p) \sin \theta \\ y_p&=y_e+ (r_e-r_p) \cos \theta\end{align*}
$$

where $\theta$ is the random _orientation_ sampled from the following distribution:

$$
\theta \sim U(0,2\pi)
$$

## Server

In order to handle the HTTP requests, I created a server using [Flask](https://flask.palletsprojects.com/en/3.0.x/). I created two end-points:

- `googly_eyes` to perform the actual image manipulation
- `identify_faces` to return information about the detect faces including bounding boxes and eye positions for debugging purposes

In both cases, the photo is stored in the body of the POST request. All image processing and manipulation was then performed in memory, so no images are ever stored to disk on the server.

For the `googly_eyes` end-point, additional parameters for the eye and pupil size can be included in the body. This allows the client to adjust these settings to personal preference.

For the `identify_faces` end-point, the face positions are returned as JSON.

When running the server for production, I used the [Gunicorn](https://gunicorn.org/) web server. The Python dependencies are managed using [Poetry](https://python-poetry.org/) and the server is encapsulated within a docker container.

## Dashboard

To allow the user to interact with the server more easily, I build a minimal dashboard using [Streamlit](https://streamlit.io/) which allows the user to:

- Upload a photo
- Adjust parameters for the googly eyes
- Download the modified photo

![Screenshot 2024-02-05 at 16.12.39.png](images/Screenshot_2024-02-05_at_16.12.39.png)

The dashboard then performs the following steps:

- Gets the settings from the information entered by the user
- Makes the HTTP request to the server to add the googly eyes
- Displays the result and adds a download link

The dashboard is hosted in a separate Docker container, with its own smaller set of dependencies using Poetry.

## Deployment

Since the server was already contained within a Docker container, it was quite simple to convert it into an AWS Lambda function. This is fine for a proof of concept, but is quite slow due to the server being quite underpowered.
