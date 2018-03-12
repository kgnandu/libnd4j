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
        T probability = (*input)(0); //(T)input0.0;
        for (int e = 1; e < input->lengthOf(); e++) {
            probability += (*input)(0);
        }
        probability /= (probability * input->lengthOf());
        //RandomLauncher<T>::fillBinomial(rng, output, lastDim, probability);
        T args[] = {(T) lastDim, probability};

        output->template applyRandom<randomOps::BinomialDistributionEx<T>>(rng, output, output, args);

        int rowCount = output->rows();
        int colCount = output->columns();
        for (int r = 0; r < rowCount; r++) {
            float maxV = (*output)(r, 0);
            for (int c = 1; c < colCount; c++) {
                if ((*output)(r, c) > maxV) {
                    maxV = (*output)(r, c);
                }
            }
            for (int c = 0; c < colCount; c++) {
                (*output)(r, c) = nd4j::math::nd4j_floor(nd4j::math::nd4j_exp((*output)(r, c) - maxV));
            }
        }
  
        return ND4J_STATUS_OK;
    }
    template int multiNomialFunctor(nd4j::random::RandomBuffer* rng, NDArray<float>* input, NDArray<float>* output, int lastDim, int seed);
    template int multiNomialFunctor(nd4j::random::RandomBuffer* rng, NDArray<float16>* input, NDArray<float16>* output, int lastDim, int seed);
    template int multiNomialFunctor(nd4j::random::RandomBuffer* rng, NDArray<double>* input, NDArray<double>* output, int lastDim, int seed);

}
}
}