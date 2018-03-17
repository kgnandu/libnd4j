//
// Created by raver119 on 01/11/17.
//

#if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__square)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(square, 1, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            T extras = (T) 2.0f;
            input->template applyTransform<simdOps::Pow<T>>(output, &extras);

            return ND4J_STATUS_OK;
        }
    }
}

#endif