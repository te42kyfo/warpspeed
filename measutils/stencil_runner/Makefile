


LDFLAGS  :=  -lnvrtc -L$(CUDA_HOME)/lib64 -lcuda -lcudart
CCFLAGS  :=  -I$(CUDA_HOME)/include -std=c++2a


cuda_stencil_runner.so: cuda_stencil_runner.cpp
	g++ $< -fPIC -shared -o $@ $(LDFLAGS) $(CCFLAGS)

hip_stencil_runner.cpp: cuda_stencil_runner.cpp
	hipify-perl $< > $@

hip_stencil_runner.so: hip_stencil_runner.cpp
	hipcc $< -fPIC -shared -o $@ -I/opt/rocm/include/hip

cuda_stencil_runner_test: cuda_stencil_runner.cpp
	g++ $< -o $@ $(LDFLAGS) $(CCFLAGS)

hip_stencil_runner_test: hip_stencil_runner.cpp
	hipcc $< -o $@ -I/opt/rocm/include/hip

clean:
	rm cuda_stencil_runner.so
	rm hip_stencil_runner.so
	rm cuda_stencil_runner_test
