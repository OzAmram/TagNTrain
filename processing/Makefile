
#A crappy MakeFile to make the Nsubjettiness python library
#Warning : You need to have compiled fastjet-contrib with CXXFLAGS=-fPIC !!!!!
# eg:  ./configure --fastjet-config=fastjet-config CXXFLAGS=-fPIC

makePyLib: NsubjettinessWrapper.i NsubjettinessWrapper.cc
	swig -c++ -python NsubjettinessWrapper.i
	g++ -fPIC -c NsubjettinessWrapper.cc  NsubjettinessWrapper_wrap.cxx `python3-config --cflags`  `fastjet-config --cxxflags --libs --plugins` -lNsubjettiness -I.
	g++ -fPIC -shared NsubjettinessWrapper.o NsubjettinessWrapper_wrap.o `fastjet-config --libs --plugins` -lNsubjettiness -o _NsubjettinessWrapper.so

.PHONY: clean

clean:
	rm -rf *.so *.o NsubjettinessWrapper_wrap.cxx NsubjettinessWrapper.py
