# use a Google maintained base image hosted in 
# Google's container registry
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-2.py310

#Upgrade pip
RUN pip install --upgrade pip

#1 make src directory 
COPY environment.yml environment.yml

RUN conda update conda
RUN conda env update -n base -f environment.yml

COPY master_audio.py-0.2.1-py3-none-any.whl master_audio.py-0.2.1-py3-none-any.whl

#RUN mkdir /master_audio

#COPY . /master_audio 

#WORKDIR /master_audio/

RUN pip install master_audio.py-0.2.1-py3-none-any.whl

COPY run_clf.py run_clf.py


#RUN export PYTHONPATH=/master_audio/
#CD into master_audio

#RUN rm -rf /master_audio

#execute code
ENTRYPOINT ["python", "run_clf.py"]