+++
title = 'CI for your CV'
date = 2024-08-12T08:46:08Z
tags=['development']
+++

I have recently started to use LaTeX again to write my CV after a long hiatus. A lot has changed since I wrote my PhD thesis using LaTeX back in 2020, both in terms of the available tooling and my knowledge of modern software development practices. Therefore I made some significant updates to my LaTeX workflow which I will describe below.

## Switching to VS code as an IDE

During my PhD, I primarily used the following LaTeX specific editors:

- [TeXStudio](https://www.texstudio.org/) on Windows
- [Texifier](https://www.texifier.com/) (formerly TeXpad) on the Mac

The main reasons why I chose to use a specific LaTeX IDE were:

1. They automatically work out the build process for you, so you can just press build and it spits out a PDF
2. They provide LaTeX specific features such as assembling the document hierarchy for easy navigation
3. That was what everyone else in my group did

To be honest, I never really loved any of these options because they weren’t particularly good _text_ editors with odd keybindings and limited theming options. With the exception of Texifier on the Mac, they tend to be ugly GTK apps and are not particularly enjoyable to use.

The obvious solution to this is to use the IDE I use everyday for Python development which is VS Code. To get LaTeX specific functionality like syntax highlighting, you can use the [LaTeX Workshop extension](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop). When I tried using this extension back in 2020, I found it to be lacking in features. However, it has matured considerably since then and IMO gives just as good an experience as a LaTeX specific editor.

To automatically work out the build process, you can use the [latexmk](https://mg.readthedocs.io/latexmk.html) Perl script in place of `pdflatex` etc. This gives you a simple _just build my document_ command which you can use with any editor you like.

## Pre-commit hooks

### Spell check

Once you’ve written your document, you obviously want to check for spelling mistakes. When I wrote my PhD thesis, I did this completely manually which is clearly not a full proof strategy. I knew there must be some tool for this, and I found the [cspell](https://github.com/streetsidesoftware/cspell-cli) utility from the makers of the [CSpell VS code extension](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker).

To add a custom LaTeX dictionary for `cspell` so that it doesn't flag macros, environments etc, you can add the following lines to the `cspell.json` configuration file:

```json
{
  "import": ["@cspell/dict-latex/cspell-ext.json"]
}
```

Since I wanted to perform these checks automatically, I configured a pre-commit hook to run on all LaTeX files by adding the following to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/streetsidesoftware/cspell-cli
    rev: v8.11.0
    hooks:
      - id: cspell
        entry: cspell-cli
        language: node
        types: [file]
        files: \.(tex|sty|cls)$
```

### Formatting

I’ve now become accustomed to formatting all of my Python code using `black`, because it improves legibility and removes whitespace changes from git diffs. Fortunately there is a similar formatter available for LaTeX in [latexindent.pl](https://github.com/cmhughes/latexindent.pl). This can also be configured as a pre-commit hook as follows:

```yaml
repos:
  - repo: https://github.com/cmhughes/latexindent.pl
    rev: V3.23.4
    hooks:
      - id: latexindent
        entry: latexindent.pl
        args: ["-wd", "-s", "-c", ".latexindent"]
        language: perl
        types: [file]
        files: \.(tex|sty|cls)$
```

where the following flags have been set:

- `-wd` ensures it writes any changes to file
- `-s` suppresses any output
- `-c` sets the directory for temporary files (so that you can add it to `.gitignore`)

Unfortunately, unlike `black`, this does not ship as a binary and requires `perl` to be installed on your system. I had to install some additional packages on my Mac to get this to work:

```yaml
cpan -i YAML::Tiny File::HomeDir Unicode::GCString
```

A more robust solution is use a devcontainer as described below.

## Using a devcontainer

I’m a strong advocate for [using devcontainers to standardise your environment]({{< ref "/posts/devcontainers" >}}), and writing LaTeX documents is no different. There are many pre-built devcontainer definitions such as [qdm12/latexdevcontainer](https://github.com/qdm12/latexdevcontainer) or [a-nau/latex-devcontainer](https://github.com/a-nau/latex-devcontainer), but I decided to write my own, because I wanted to a bit more control of the LaTeX installation. Once I had got this working reliably, I was able to make use of GitHub codespaces. This gave me something akin to [Overleaf](https://www.overleaf.com/), but using the more familiar VS Code editor.

### Installing LaTeX dependencies

To install LaTeX on a Debian-based OS, you need to install some additional system dependencies:

```docker
# Install required system packages for LaTeX
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  ghostscript \
  gnupg \
  perl \
  && rm -rf /var/lib/apt/lists/*
```

To use `latexindent.pl`, you also need to install some additional `perl` packages:

```docker
# Install dependencies needed by latexindent
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  libunicode-linebreak-perl\
  libyaml-tiny-perl \
  libfile-homedir-perl \
  && rm -rf /var/lib/apt/lists/*
```

### Minimal LaTeX installation

The main challenge I ran into was configuring a minimal LaTeX installation, with only those packages required for my document. To install TeXLive with minimal packages you can use the following bash script:

```bash
#!/bin/bash

set -e

echo "==> Install TeXLive"
mkdir -p ./texlive
MIRROR_URL="$(curl -w "%{redirect_url}" -o /dev/null https://mirror.ctan.org/)"
curl --output-dir ./texlive -OL "${MIRROR_URL}systems/texlive/tlnet/install-tl-unx.tar.gz"
curl --output-dir ./texlive -OL "${MIRROR_URL}systems/texlive/tlnet/install-tl-unx.tar.gz.sha512"
curl --output-dir ./texlive -OL "${MIRROR_URL}systems/texlive/tlnet/install-tl-unx.tar.gz.sha512.asc"
mkdir -p ./texlive/installer
tar --strip-components 1 -zxf ./texlive/install-tl-unx.tar.gz -C ./texlive/installer
sudo ./texlive/installer/install-tl -profile=./texlive.profile

echo "==> Clean up"
rm -rf \
  /usr/local/texlive/texdir/install-tl \
  /usr/local/texlive/texdir/install-tl.log \
  ./texlive
```

which can then be called within your `Dockerfile`. The `texlive.profile` file is used to configure what will be installed. Here is my version which only installs the default LaTeX packages, and no documentation or source code:

```bash
selected_scheme scheme-infraonly
TEXDIR /usr/local/texlive/
TEXMFCONFIG ~/.texlive/texmf-config
TEXMFHOME ~/texmf
TEXMFLOCAL /usr/local/texlive/texmf-local
TEXMFSYSCONFIG /usr/local/texlive/texmf-config
TEXMFSYSVAR /usr/local/texlive/texmf-var
TEXMFVAR ~/.texlive/texmf-var
option_doc 0
option_src 0
collection-latex 1
```

Unfortunately I couldn’t find a way to specify a set of LaTeX packages to be installed. As a workaround, I created a `texlive-packages.txt` file with a list of package names as follows:

```
tabularx
graphicsx
...
```

To then install only these packages, you add the following to your `post_start.sh` file:

```bash
cat texlive-packages.txt | sed -re '/^#/d' | xargs sudo tlmgr install
```

which will install only the specified packaged using the `tlmgr` package manager. Obviously this does not pin the package versions, but at least you don't have to install every LaTeX package under the sun for a simple document.

## Continuous integration

Since my source code was stored in GitHub and I could write my document using GitHub codespaces, I thought it would be nice to store the generated PDFs there as well. This last step is probably a bit overkill for most people, but I found it to be useful for managing different versions of my CV when applying for jobs.

I built a workflow with GitHub actions making use of the handy [LaTeX GitHub action](https://github.com/marketplace/actions/github-action-for-latex), which runs the following jobs on all branches:

- Pre-commit checks
- Builds the document to check that it still compiles

Finally, once I’m happy with a specific version, I add a Git tag to trigger a new release with the PDF as an artifact.

```yaml
name: Build LaTeX document

on:
  push:
    branches:
      - "*"
    tags: ["v*.*.*"]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      # Install Perl for latexindent
      - uses: shogo82148/actions-setup-perl@v1
        with:
          install-modules: YAML::Tiny File::HomeDir
      - uses: actions/setup-python@v3
      - uses: pre-commit/action@v3.0.0
  build:
    needs: pre-commit
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
      - name: Compile LaTeX document
        uses: xu-cheng/latex-action@v3
        with:
          root_file: document.tex
      - name: Upload PDF artifact
        uses: actions/upload-artifact@v3
        with:
          name: PDF
          path: document.pdf
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref_type == 'tag'
    permissions:
      contents: write
    steps:
      - name: Download PDF artifact
        uses: actions/download-artifact@v3
        with:
          name: PDF
      - name: Create release with PDF
        uses: softprops/action-gh-release@v1
        with:
          files: document.pdf
          fail_on_unmatched_files: true
```
