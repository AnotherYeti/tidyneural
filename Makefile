UNAME := $(shell uname)

bobjects = NNetwork.o NWeight.o Population.o Actor.o CInterfaceActor.o

btestobjects = main.o xor.o

BUILDDIR = build/

objects := $(addprefix $(BUILDDIR), $(bobjects))

testobjects := $(addprefix $(BUILDDIR), $(btestobjects))

CFLAGS = -arch i386 -arch x86_64

all : lib

test : lib $(testobjects)
	g++ $(CFLAGS) -O3 -g -Wall -o test -L. -lNeural $(testobjects)

build/%.o : src/%.cpp
	g++ $(CFLAGS) -O3 -g -c -Wall $< -o $@

lib : $(objects)
	g++ $(CFLAGS) -O3 -g -Wall -shared -o libNeural.so $(objects)
	g++ -arch i386 -O3 -g -Wall -bundle -o Neural.bundle $(objects)

clean :
	rm -f build/*.o libNeural.so *.bundle
