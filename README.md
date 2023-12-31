# [Alex Haslam's Personal Website](https://alxhslm.github.io/)

## Installation

1. Install [Hugo](https://gohugo.io/) on your machine:

```
brew install hugo
```

2. Clone this repo using:

```
git clone git@github.com:alxhslm/alxhslm.github.io.git
```

3. From the root of the repo, install the [Blowfish](https://blowfish.page/) theme using:

```
git submodule add -b main https://github.com/nunocoracao/blowfish.git themes/blowfish
```

4. Install [`pre-commit`](https://pre-commit.com/) hooks by running:

```
pip install pre-commit
pre-commit install
```

## Local development

Run the server locally using:

```
hugo server --buildDrafts
```

Once you're happy with your changes, build the site using:

```
hugo
```

and commit the result.
