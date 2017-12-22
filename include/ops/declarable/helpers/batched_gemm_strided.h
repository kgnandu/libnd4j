//
//  @author raver119@gmail.com
//

#include <NDArray.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void _bgemms(NDArray<T> *A, NDArray<T> *B, NDArray<T> *C, NDArray<T> *alphas, NDArray<T> *betas);
        }
    }
}