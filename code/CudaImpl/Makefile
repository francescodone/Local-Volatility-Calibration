CXX        = nvcc

INCLUDES    += -I ../include
GPU_OPTS   = -D lgWARP=5

SOURCES_CPP =ProjectMain.cu ProjHelperFun.cu ProjCoreOrig.cu 
SRC1 = ProjectMain.cu
HELPERS     =ProhHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h
EXECUTABLE  =runproject

default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(GPU_OPTS) $(INCLUDES) -c $@ $<

compile: $(EXECUTABLE)
$(EXECUTABLE): $(SOURCES_CPP)
	$(CXX) $(INCLUDES) -o $(EXECUTABLE) $(SOURCES_CPP)


run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE)
