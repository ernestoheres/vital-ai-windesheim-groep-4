FROM ubuntu:24.04
#FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 #for Nvidia GPUs only

RUN apt-get update && apt-get install -y python3-full python3-pip\
 graphviz fonts-dejavu-core texlive-xetex pandoc\
 && rm -rf /var/lib/apt/lists/*
RUN pip config set global.break-system-packages true
RUN pip install jupyter ipywidgets ipykernel pandas numpy scikit-learn\
 seaborn matplotlib statsmodels torch==2.5 neuralprophet minisom\
 plotly-resampler openpyxl geopandas folium palmerpenguins\
 ucimlrepo xgboost dtreeviz ucimlrepo 

WORKDIR "/tmp"