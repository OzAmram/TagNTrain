 /* NsubjettinessWrapper.i */

%include "typemaps.i"
%include "pyabc.i"
%include "std_string.i"
%include "std_vector.i"
%include "carrays.i"


namespace std {
   %template(vectord) vector<double>;
};

%module NsubjettinessWrapper
 %{
 /* Put header files here or function declarations like below */
#include "NsubjettinessWrapper.h"
 %}


%include "NsubjettinessWrapper.h" 
