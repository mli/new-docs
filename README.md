# README

## What's new

This is a total new design of the mxnet webiste. Major changes includes:

1. used a relative simple build pipeline with a new theme. Removed a bunch of html/js hacks.
2. easier to add contents without using HTML. such as adding a new card on the frontpage:

   ```rst
   .. container:: mx-card

      :card-title:`GluonCV`
      :card-text:`A deep learning toolkit for computer vision.`
      :card-link:`https://gluon-cv.mxnet.io`
   ```
   
   or install options:
   
   ```rst
   .. container:: opt-group

      :title:`Platform:`
      :opt:`Linux`
      :opt:`Macos`
      :opt:`Windows`
      :opt:`Cloud`
   ```
3. reorgized all documents by following tensorflow and pytorch. All tutorials go to develop/ and APIs go to api/. 
4. Any tutorial under the develop/ page should be evaluated to get real outputs during build.
5. Enabled right TOC, it's useful to navigate within the current page. 
6. Each API function has a single page, instead of stacking all of them in a long page. All APIs can be accessed on the right toc panel. 
7. Added disqus links so that readers can ask questions easily 

## Setup

You need to have CUDA 9.2 installed. If you prefer to skip evaluation on GPUs, you can change `mxnet-cu92` into `mxnet` in the `environment.yml` file. 

Run the following commands to setup the environment.

```bash
git submodule update --init --recursive
conda env create -f environment.yml
source activate mxnet-docs
```

## Build the docs

To build without testing the notebooks:

```bash
make EVAL=0
```

To build with testing the notebooks (requires GPU):

```bash
make
```

The build docs will be available at `build/_build/html`. 

Each build may take a few minutes even without evaluation. To accelearte it, we can use one of the following ways:

1. open `build/conf.py`, add the folders you want to skip into `exclude_patterns`, such as `exclude_patterns = ['templates', 'sphinx_materialdesign_theme', 'api', 'develop', 'blog']`. 
2. move the files into a different folder, such as `mv api /tmp/`, and then `make clean`.
