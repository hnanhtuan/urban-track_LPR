CC = g++
OUTPUT = main
CFLAGS = -Wall -shared -std=c++11

# OPENCV
INCLUDE = `pkg-config --cflags opencv`  
LIBS = `pkg-config --libs opencv` -larmadillo -ltesseract -llept

# CAFFE
CAFFE_PATH = /home/anhxtuan/Downloads/caffe-master
INCLUDE += -I${CAFFE_PATH}/include
LIBS += -L${CAFFE_PATH}/build/lib -lcaffe -lboost_system -lglog -lboost_filesystem -lboost_serialization -ltbb

# CUDA
CUDA_INSTALL_PATH = /usr/local/cuda
INCLUDE += -I$(CUDA_INSTALL_PATH)/include
LIBS += -L$(CUDA_INSTALL_PATH)/lib64   -lpthread -lcudart

CPP_FILES = $(wildcard src/*.cpp)
OBJS := $(addprefix src/,$(notdir $(CPP_FILES:.cpp=.o)))
OBJS += ${OUTPUT}.o

all: ${OUTPUT}

${OUTPUT}: ${OBJS}
	${CC} -o ${OUTPUT} ${OBJS} ${LIBS}

${OUTPUT}.o: ${OUTPUT}.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c ${OUTPUT}.cpp
	
src/%.o: src/%.cpp
	${CC} ${INCLUDE} ${CFLAGS} -c -o $@ $<

clean:
	rm -f $(OBJS) $(OUTPUT)
