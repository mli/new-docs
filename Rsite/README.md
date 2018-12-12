# README

Follow these steps to build the MXNet R website.

## Setup 

You need to have necessary software installed, and Jupyter Notebook with R kernel and notedown compatibility (to compile tutorials).

1) install MXNet R version:
http://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=R&processor=CPU 

2) Install other necessary R packages:
Start R in your terminal (Note: cannot use R studio here to ensure Jupyter kernel can access these packages as well):

install.packages(c('mxnet', 'repr', 'IRdisplay', 'evaluate', 'crayon', 'pbdZMQ', 'devtools', 'uuid', 'digest'))

3) Install Jupyter Notebook with R kernel and notedown compatibility:
https://www.datacamp.com/community/blog/jupyter-notebook-r 
https://github.com/aaren/notedown 

Add the following to your Juypter config file:  c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'

3) Install the necessary mxtheme python package (used for website style):

pip install https://github.com/mli/mx-theme/tarball/master

4) Install sphinx & pandoc (used for processing Notebooks with Sphinx):  
http://pandoc.org/installing.html

5) You will also need to have the .Rd documentation files already generated for the 'mxnet' R package.
Up-to-date R documentation files can be obtained by building MXNet from the source:

https://github.com/apache/incubator-mxnet 

These files will then be located in the directory: PATH:/mxnet/R-package/man/ 

Copy all of them into subdirectory named: source/api/man/   (replacing existing files as needed)  

You can check out: scripts/Rdoc2SphinxDoc.R to see locations of all files that the sphinx-documentation-generation script depends on.

6) For the tutorials, in addition to the .md Notebook files, you must have the following files available in the same directory:  
data/train.csv, data/test.csv (for DigitsClassification tutorial)
Inception/Inception_BN-symbol.json (for ClassifyImageWithPretrainedModel tutorial)

You should also ensure the file tutorial/index.rst lists all the tutorials you wish to include on the website (and in the proper order). 

If adding new object files that future tutorials will depend on, you must update the makefile to ensure they get copied into the mxnetRtutorials/ subdirectory which is subsequently zipped and made available for user-download (so they can run the Jupyter notebooks themselves).  


## Commands to build the MXNet-R website: (homepage will appear in: ./build/index.html)

make clean
make


Detailed descriptions: 

# build documentation:
cd ./source/
Rscript ./scripts/Rdoc2SphinxDoc.R

# build tutorials:
cd ./source/tutorial/
bash scripts/convertNotebooks.sh

# build sphinx page:
cd ./ # (must be in home directory containing source/ and build/ subdirectories)
sphinx-build -b html source/ build/


## TODOs:

In ClassifyImageWithPretrainedModel.md tutorial:
    - model-zoo link should be updated to new MXNet site ("Model Zoo"); as should link for downloading pre-trained network ("this link": http://data.mxnet.io/mxnet/data/Inception.zip). 

In Symbol.md tutorial:
    - Link (Symbolic Configuration and Execution in Pictures) should be updated to point at new MXNet website.

In index.rst R homepage: 
- Link to main mxnet site maybe should be updated to Mu's version. 
- Installation instructions

In tutorial/index.rst page:
- TODO on how to download & run notebooks needs to be addressed.

