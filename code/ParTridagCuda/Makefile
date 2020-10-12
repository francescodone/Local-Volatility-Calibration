CXX        = nvcc

SOURCES_CPP =TestCudaTridag.cu
HELPERS     =TridagPar.h TridagKernel.cu.h
EXECUTABLE  =test_tridag


default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)
$(EXECUTABLE): 
	$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

