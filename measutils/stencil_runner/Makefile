


HIP_HOME :=  /opt/rocm

LDFLAGS  :=  -lnvrtc -L$(CUDA_HOME)/lib64 -lcuda -lcudart -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti  -lnvidia-ml -lnvperf_host -lnvperf_target $(shell python3-config --ldflags)
CUDAFLAGS  :=  -O3 -I$(CUDA_HOME)/include -std=c++2a $(shell python3-config --includes) -I$(CUDA_HOME)/extras/CUPTI/include -Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/

HOSTNAME = $(shell hostname)

HIPFLAGS := -O3 -std=c++2a -I/opt/rocm/include/hip $(shell python3-config --includes) -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lrocm_smi64 -lhsa-runtime64 $(shell python3-config --ldflags)

DEPS := ../gpu_metrics/rocm_metrics/rocm_metrics.hpp

cuda_stencil_runner: cuda_stencil_runner.cpp
	echo $(HOSTNAME)
	g++ $< -fPIC -shared -o $@.$(HOSTNAME).so $(LDFLAGS) $(CUDAFLAGS)

cuda_conv_runner.so: cuda_conv_runner.cpp
	g++ $< -fPIC -shared -o $@ $(LDFLAGS) $(CUDAFLAGS)

hip_stencil_runner.cpp: cuda_stencil_runner.cpp
	hipify-perl $< > $@

hip_rtc.hpp: cuda_rtc.hpp
	hipify-perl $< > $@


hip_stencil_runner: hip_stencil_runner.cpp hip_rtc.hpp
	echo $(shell hostname)
	hipcc $< -fPIC -shared  -o $@.$(shell hostname).so $(HIPFLAGS)

hip_conv_runner.cpp: cuda_conv_runner.cpp
	hipify-perl $< > $@

hip_conv_runner.so: hip_conv_runner.cpp
	hipcc $< -fPIC -shared  -o $@ $(HIPFLAGS)


cuda_stencil_runner_test: cuda_stencil_runner.cpp
	g++ $< -o $@ $(LDFLAGS) $(CUDAFLAGS)

hip_stencil_runner_test: hip_stencil_runner.cpp $(DEPS)
	hipcc $< -o $@ $(HIPFLAGS) -ldl -DNO_PYTHON

clean:
	-@rm *.so
	-@rm hip_stencil_runner.cpp
	-@rm cuda_stencil_runner_test
	-@rm hip_stencil_runner_test
