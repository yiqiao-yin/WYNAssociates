from jupyter/datascience-notebook:ubuntu-20.04

USER root

RUN sudo apt update && \
    apt-get install python3-openslide -y

USER 1000

RUN conda install git
RUN pip install pandas
RUN pip install openpyxl

RUN pip install openslide-python
RUN pip install opencv-python

RUN conda install -y -c pytorch torchvision

RUN pip install lightgbm

RUN pip install plotly
RUN pip install plotly-express

RUN pip install yfinance
RUN pip install ta

RUN pip install tensorflow
RUN pip install tensorflow-datasets

RUN pip install pydot
RUN pip install graphviz

# RUN pip install javabridge bioformats

# RUN pip install SimpleITK geojson shapely progressbar

# RUN pip install pyinstaller
# RUN pip install dash plot_common shapes_to_segmentations trainable_segmentation

# RUN conda install lazypredict