all:
	nvcc -dc CUDACompress.cu parallelFunctions.cu kernel.cu GPUWrapper.cu
	nvcc *.o -o CUDA_compress
	rm -rf *.o

clean:
	if [ -a CUDA_compress ]; then rm -f CUDA_compress; fi;
