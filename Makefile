lang_flags = -std=c++0x -Werror -Wall
#lang_flags = -Werror -Wall
#CXXFLAGS += $(lang_flags) -g -save-temps 
CXXFLAGS += $(lang_flags) -g
incl_path = -Iinclude -I/opt/opencv/include
CPPFLAGS += -MMD -MP $(incl_path) 
#CPPFLAGS += -MMD -MP $(incl_path) -D_GLIBCXX_DEBUG
LIB_SRC = $(wildcard src/*.cpp)
TEST_SRC = $(wildcard test/*.cpp)
#hardcode location of opencv libs into executables...
LDFLAGS = -Wl,-R/opt/opencv/lib
LDLIBS = -L/opt/opencv/lib -lopencv_core -lopencv_contrib -lopencv_calib3d \
	-lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui \
	-lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree

all: objrec run_tests TAGS
.PHONY: all

TAGS: objrec test/test
	ctags -eR --languages=-Java main.cpp include src test OpenCV-2.4.0 

objrec: main.cpp src/libobjrec.a
	g++ $(LDFLAGS) $(CXXFLAGS) $(CPPFLAGS) -o $@ main.cpp -Lsrc -lobjrec $(LDLIBS)

-include objrec.d

src/libobjrec.a: $(LIB_SRC:%.cpp=%.o)
	$(AR) $(ARFLAGS) $@ $^

-include $(LIB_SRC:%.cpp=%.d)

check-syntax:
	g++ -o nul -S ${lang_flags} ${incl_path} ${CHK_SOURCES}

test/test: $(TEST_SRC:%.cpp=%.o) src/libobjrec.a
	g++ $(LDFLAGS) -o $@ $^ $(shell gtest-config  --ldflags  --libs) $(LDLIBS)

-include $(TEST_SRC:%.cpp=%.d)

run_tests: test/test
	./test/test
.PHONY: run_tests

clean:
	rm -f main.o objrec objrec.d TAGS
	rm -f $(LIB_SRC:%.cpp=%.o) $(LIB_SRC:%.cpp=%.d) src/libobjrec.a  
	rm -f $(TEST_SRC:%.cpp=%.o) $(TEST_SRC:%.cpp=%.d) test/test
	rm -f *.ii *.s
.PHONY: clean
