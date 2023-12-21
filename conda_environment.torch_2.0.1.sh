#!/bin/bash

PYTHON_VERSION=3.10
CUDA_VERSION=11.7.99
CUDNN_VERSION=8.5.0.96
PYTORCH_VERSION=2.0.1

# #################### get env directories
# CONDA_ROOT
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
echo "CONDA_CONFIG_ROOT_PREFIX= ${CONDA_CONFIG_ROOT_PREFIX}"
get_conda_root_prefix() {
  TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
  TMP_POS=$((TMP_POS-1))
  if [ $TMP_POS -ge 0 ]; then
    echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
  else
    echo ""
  fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
  echo "CONDA_ROOT= ${CONDA_ROOT}, not exists, exit"
  exit 1
fi


find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

CONDA_NEW_ENV=llm-cuda_${CUDA_VERSION}-cudnn_${CUDNN_VERSION}-torch_${PYTORCH_VERSION}

if find_in_conda_env "^${CONDA_NEW_ENV}\s" ; then
  echo "Found ${CONDA_NEW_ENV}, skip create"
  conda activate ${CONDA_NEW_ENV}
else
  conda create -n ${CONDA_NEW_ENV}  -y
  conda activate ${CONDA_NEW_ENV}
fi


conda info

conda install python=${PYTHON_VERSION} -y && \
    conda install conda-verify conda-build mkl-include cmake==3.27.0 ninja==1.11.1 -c anaconda -y && \
    conda install nodejs=18.12.1 -c conda-forge -y
#    conda clean -afy

pip install pandas  \
	cachetools==4.2.2 tensorboardX==2.5 \
	matplotlib==3.5.3 pyproject-toml==0.0.11 xlrd==2.0.1 tqdm==4.62.2

# pip install --no-cache-dir numba==0.48.0

pip install torch==${PYTORCH_VERSION} nvidia_cuda_runtime_cu11==${CUDA_VERSION} nvidia_cudnn_cu11==${CUDNN_VERSION} \
  transformers==4.31.0 cpm_kernels gradio mdtex2html sentencepiece langchain==0.0.174 protobuf==3.20.3 tabulate==0.9.0 \
  datasets==2.12.0 accelerate==0.21.0 trl==0.4.4 transformers_stream_generator==0.0.4 sentence_transformers==2.2.2
  
pip install git+https://github.com/huggingface/peft.git@13e53fc

pip install -U "notebook[all]==6.5.5" ipython==8.14.0 traitlets==5.9.0 ipyparallel jupyter_lsp jupyter_nbextensions_configurator jupyter_contrib_nbextensions ipywidgets
#usual installation
jupyter nbextension enable --py widgetsnbextension
#you are my saver!
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# check torch GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"

# check library versions
echo "[NumPy]"
python -c "import numpy as np; print(np.__version__)"
echo "[Torch]"
python -c "import torch; print(torch.__version__)"
# python -c "import torchvision; print(torchvision.__version__)"
