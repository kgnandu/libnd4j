//
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/multinomial.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <vector>
#include <memory>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int multiNomialFunctor(nd4j::random::RandomBuffer* rng, NDArray<T>* input, NDArray<T>* output, int lastDim, int seed) {

        if (seed) {
            NativeOps native;

            native.reSeedBuffer(nullptr, (long)seed, rng);
        }
        //if (newRng )
        if (rng == nullptr)
            return ND4J_STATUS_BAD_RNG;

  
        return ND4J_STATUS_OK;
    }
    template int multiNomialFunctor(nd4j::random::RandomBuffer* rng, NDArray<float>* input, NDArray<float>* output, int lastDim, int seed);
    template int multiNomialFunctor(nd4j::random::RandomBuffer* rng, NDArray<float16>* input, NDArray<float16>* output, int lastDim, int seed);
    template int multiNomialFunctor(nd4j::random::RandomBuffer* rng, NDArray<double>* input, NDArray<double>* output, int lastDim, int seed);

}
}
}