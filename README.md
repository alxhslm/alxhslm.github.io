# [Alex Haslam's Personal Website](https://alxhslm.github.io/)

## Getting started

### Using the devcontainer

The easiest way to get started is to open the `website.code-workspace` in VS code, and then select the "Reopen in container" option. This will automatically set up an environment for you with Hugo and `pre-commit` hooks.

### Setting up your machine

If you wish to work locally on your machine, then follow theses steps:

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
git submodule update --init
```

4. Install [`pre-commit`](https://pre-commit.com/) hooks by running:

```
pip install pre-commit
pre-commit install
```

## Testing the website locally

Run the server locally using:

```
hugo server --buildDrafts
```

Once you're happy, commit and push your changes. The site will automatically get rebuilt and deployed from the `main` branch using GitHub actions.
