.PHONY: default
default: all ;

CC        := g++
NVCC      := nvcc
SRCDIR    := src
BUILDDIR  := build
LIBRARY   := libSPMatrix.so
STATICLIB := libSPMatrix.a
TARGETDIR := bin
LIBDIR    := lib
TESTDIR   := test

NVSRCEXT  := cu
SRCEXT    := cpp
SOURCES   := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
NVSOURCES := $(shell find $(SRCDIR) -type f -name *.$(NVSRCEXT))
OBJECTS   := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
NVOBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(NVSOURCES:.$(NVSRCEXT)=.o))
CFLAGS    := -std=c++14 -O3 -fPIC -fopenmp -Wall -Wno-unused-function
NVCFLAGS  := -arch=sm_61 -I. -rdc=true -Xcompiler -fPIC -Xcompiler -O3
NVLFLAGS  := -arch=sm_61 -dlink -Xcompiler -fPIC
SO_FLAGS  := -shared
INC       := -I include

library: $(LIBDIR)/$(LIBRARY)

all: $(LIBDIR)/$(LIBRARY)

static: $(LIBDIR)/$(STATICLIB)

$(LIBDIR)/$(STATICLIB) : $(BUILDDIR)/cuda_link.o $(OBJECTS) $(NVOBJECTS)
	@mkdir -p $(LIBDIR)
	@ar rsv $@ $^

testing: $(LIBDIR)/$(LIBRARY)
	@mkdir -p $(TARGETDIR)
	@echo " $(CC) $(CFLAGS) $(INC) $(TESTDIR)/TestMatrix.cpp -o bin/TestMatrix"
	$(CC) $(CFLAGS) $(INC) -L$(LIBDIR) -lSPMatrix -L/opt/cuda/lib64 -lcuda -lcudart $(TESTDIR)/TestMatrix.cpp -o bin/TestMatrix

$(LIBDIR)/$(LIBRARY): $(BUILDDIR)/cuda_link.o $(OBJECTS) $(NVOBJECTS)
	@mkdir -p $(LIBDIR)
	$(CC) $(SO_FLAGS) -Wl,-soname,$(LIBRARY) -o $@ $^ -lc

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(NVSRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo "$(NVCC) $(NVCFLAGS) $(INC) -c -o $@ $<"
	$(NVCC) $(NVCFLAGS) $(INC) -c -o $@ $<
	
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/cuda_link.o: $(NVOBJECTS)
	@mkdir -p $(BUILDDIR)
	@echo "$(NVCC) $(NVLFLAGS) -o $@ $<"
	$(NVCC) $(NVLFLAGS) -o $@ $<

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(LIBDIR) $(TARGETDIR)"
	$(RM) -r $(BUILDDIR) $(LIBDIR) $(TARGETDIR)
