FROM nvcr.io/nvidia/pytorch:21.02-py3
LABEL org.opencontainers.image.authors="h.kuijf@umcutrecht.nl"

RUN mkdir -p /home

WORKDIR /home

RUN python -m pip install -U pip

COPY requirements.txt /home/
RUN python -m pip install -r requirements.txt

COPY elastix-5.0.1-linux /home/elastix-5.0.1-linux/
COPY example_data /home/example_data/
COPY model_RCNN.pt model_UNet32.pt process.py process-lacunes.py run.sh /home/

RUN mkdir /home/input_data/ /home/input_data/lacunes/ /home/output_data/ /home/output_data/lacunes/ \ 
 && chmod -R 777 /home \
 && chmod u+x /home/elastix-5.0.1-linux/bin/elastix \
 && chmod u+x /home/elastix-5.0.1-linux/bin/transformix \
 && chmod u+x /home/run.sh 


ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=mixlacune

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=10G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=10G