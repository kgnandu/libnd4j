//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int lrnFunctor(NDArray<T>* input, NDArray<T>* output, int depth, T bias, T alpha, T beta) {

        T dividor;

        int totalLength = input->lengthOf();
        int lastDim = input->sizeAt(-1);
        int chunkCount = totalLength / lastDim;

        for (int c = 0; c < chunkCount; c++) {
            for (int e = 0; e < lastDim; e++) {
                int begin = nd4j::math::nd4j_max(0, e - depth);
                int end = nd4j::math::nd4j_min(depth + e + 1, lastDim);
                T quadSum = 0;

                for (int pos = begin; pos < end; ++pos) {
                    T val = (*input)(c * lastDim + pos);
                    quadSum += val * val;
                }
                T dividor = nd4j::math::nd4j_pow(bias + alpha * quadSum, beta);
                (*output)(c * lastDim + e) = (*input)(c * lastDim + e) / dividor;
            }
        }

        return ND4J_STATUS_OK;
    }
    template int lrnFunctor(NDArray<float>* input, NDArray<float>* output, int depth, float bias, float alpha, float beta);
    template int lrnFunctor(NDArray<float16>* input, NDArray<float16>* output, int depth, float16 bias, float16 alpha, float16 beta);
    template int lrnFunctor(NDArray<double>* input, NDArray<double>* output, int depth, double bias, double alpha, double beta);

}
}
}