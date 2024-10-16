export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
export TORCH_CUDA_ARCH_LIST=$(python3 -c "import torch;a=torch.cuda.get_device_capability(0);print(str(a[0])+'.'+str(a[1]))")

export CC=gcc-11
python3 setup.py build_ext --inplace

rm -r build
