//
//  @author sgazeos@gmail.com 3/7/2018
//

#ifndef __MULTI_NOMIAL_HELPERS__
#define __MULTI_NOMIAL_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int multiNomialFunctor(nd4j::random::RandomBuffer* rng, NDArray<T>* input, NDArray<T>* output, int lastDim, int seed = 0);

}
}
}
#endif
