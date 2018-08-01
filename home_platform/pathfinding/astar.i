/* astar.i */
%module astar

%{
#define SWIG_FILE_WITH_INIT 
#include "astar.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%include "std_vector.i"

// Instantiate used templates
%template(VecFloat)       std::vector<float>;
%template(VecNode)        std::vector<Node>;
%template(VecNodePtr)     std::vector<Node*>;
%template(VecVecFloat)    std::vector< std::vector<float> >;
%template(VecVecNode)     std::vector< std::vector<Node> >;

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* weights, int dim1, int dim2)};
%include "astar.h"
%clear (float* weights, int dim1, int dim2);
