//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/batched_gemm_strided.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void _bgemms(NDArray<T> *A, NDArray<T> *B, NDArray<T> *C) {
                A->rankOf();
            };

            template void _bgemms<float>(NDArray<float> *A, NDArray<float> *B, NDArray<float> *C);
            template void _bgemms<float16>(NDArray<float16> *A, NDArray<float16> *B, NDArray<float16> *C);
            template void _bgemms<double>(NDArray<double> *A, NDArray<double> *B, NDArray<double> *C);
        }
    }
}