# Python binding docs

## Setup

You need to have CUDA 9.2 installed. If you prefer to skip evaluation on GPUs, you can change `mxnet-cu92` into `mxnet` in the `environment.yml` file.

Run the following commands to setup the environment.

```bash
conda env create -f environment.yml
source activate mxnet-docs
```

## Build the docs

To build without testing the notebooks (faster):

```bash
make EVAL=0
```

To build with testing the notebooks (requires GPU):

```bash
make
```

The build docs will be available at `build/_build/html`.

Each build may take a few minutes even without evaluation. To accelerate it, we can use one of the following ways:

1. open `build/conf.py`, add the folders you want to skip into `exclude_patterns`, such as `exclude_patterns = ['templates', 'api', 'develop', 'blog']`.
2. move the files into a different folder, such as `mv api /tmp/`, and then `make clean`.

## Check results

If you build docs in a remote machine, you can

1. start a http server: `cd build/_build/html; python -m http.server`
2. ssh to your machine with port forwarding: `ssh -L8000:localhost:8000 your_machine`
3. Open http://localhost:8000 in your local machine


## Run tutorials

In addition to view the built html pages, you can run the Jupyter notebook from a remote machine.
1. Install `notedown` plugin: `pip install https://github.com/mli/notedown/tarball/master` in remote server
2. Start Jupyter notebook `jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'` in remote server
3. ssh to your machine with port forwarding: `ssh -L8888:localhost:8888 your_machine`
4. Open http://localhost:8888 in your local machine and run the md files directly

Optionally, one can run the following to launch the notedown plugin automatically when starting jupyter notebook.
1. Generate the jupyter configure file `~/.jupyter/jupyter_notebook_config.py` if it
is not existing by run `jupyter notebook --generate-config`
2. Add `c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'` to `~/.jupyter/jupyter_notebook_config.py`
3. Simply run `jupyter notebook`

## Troubleshooting
Dependencies and the setup steps for this website are changing often. Here are some troubleshooting tips.

* You might need to update the environment for the latest modules.
```bash
conda env update -f environment.yml
```
