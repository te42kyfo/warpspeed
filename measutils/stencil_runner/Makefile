


HIP_HOME :=  /opt/rocm

LDFLAGS  :=  -lnvrtc -L$(CUDA_HOME)/lib64 -lcuda -lcudart -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti  -lnvidia-ml -lnvperf_host -lnvperf_target
CUDAFLAGS  :=  -O3 -I$(CUDA_HOME)/include -std=c++2a $(shell python3-config --includes) -I$(CUDA_HOME)/extras/CUPTI/include -Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/


HIPFLAGS := -O3 -std=c++2a -I/opt/rocm/include/hip $(shell python3-config --includes) -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64

DEPS := ../gpu_metrics/rocm_metrics/rocm_metrics.hpp

cuda_stencil_runner.so: cuda_stencil_runner.cpp
	g++ $< -fPIC -shared -o $@ $(LDFLAGS) $(CUDAFLAGS)

hip_stencil_runner.cpp: cuda_stencil_runner.cpp
	hipify-perl $< > $@

hip_stencil_runner.so: hip_stencil_runner.cpp
	hipcc $< -fPIC -shared  -o $@ $(HIPFLAGS)

cuda_stencil_runner_test: cuda_stencil_runner.cpp
	g++ $< -o $@ $(LDFLAGS) $(CUDAFLAGS)

hip_stencil_runner_test: hip_stencil_runner.cpp $(DEPS)
	hipcc $< -o $@ $(HIPFLAGS) -ldl -DNO_PYTHON

clean:
	-@rm cuda_stencil_runner.so
	-@rm hip_stencil_runner.so
	-@rm hip_stencil_runner.cpp
	-@rm cuda_stencil_runner_test
	-@rm hip_stencil_runner_test
