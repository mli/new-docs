# README

## New

Started from scratch: removed all HTML/JS hacks, and used a default theme
instead of the customized mxdoc theme. I created the mxdoc theme a few years
ago without quite understand how to make a customized theme, we need to
re-design it later.

Use two file formats: markdown and RST. Markdown is for articles and notebooks,
in default, each markdown will be converted into a jupyter notebook and be
evaluated. RST is used for sphinx-related document, such as index and API pages.

TODO:

- A new sphinx theme
- Add other languages besides python
- Port more contents from mxnet

## Setup

```bash
conda env create -f environment.yml
source activate mxnet-docs
```

## Build

To build without testing the notebooks:

```bash
make EVAL=0
```

To build with testing the notebooks (requires GPU):

```bash
make
```
