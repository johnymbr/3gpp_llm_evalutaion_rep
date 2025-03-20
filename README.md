# 3gpp_llm_evalutaion_rep

## To use Cuda in Fedora 41:

### 1. Step 

Follow link below to install cuda-toolkit on Fedora 41: \
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Fedora&target_version=41&target_type=rpm_network

### 2. Step

Install cuda-toolkit it was necessary to download and copy cusparselt lib to cuda folder: \
https://developer.nvidia.com/cusparselt-downloads?target_os=Linux&target_arch=x86_64&Distribution=Agnostic

```
sudo cp libcusparse_lt-linux-x86_64-0.7.1.0-archive/include/* /usr/local/cuda-12.6/include/
sudo cp libcusparse_lt-linux-x86_64-0.7.1.0-archive/lib/* /usr/local/cuda-12.6/lib
sudo cp libcusparse_lt-linux-x86_64-0.7.1.0-archive/lib/* /usr/local/cuda-12.6/lib64/
```

### 3. Step

It was necessary to include CUPTI into path usigin command below:

```
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.x/extras/CUPTI/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Step

It was necessary to download nccl and extract files: \ 
https://developer.nvidia.com/nccl/nccl-download
https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html

And then we ran command below to copy files from nccl folder to cuda folder.
```
sudo cp nccl_2.26.2-1+cuda12.4_x86_64/bin/* /usr/local/cuda-12.6/bin/
sudo cp nccl_2.26.2-1+cuda12.4_x86_64/include/* /usr/local/cuda-12.6/include/
sudo cp nccl_2.26.2-1+cuda12.4_x86_64/lib/* /usr/local/cuda-12.6/lib64/
```
