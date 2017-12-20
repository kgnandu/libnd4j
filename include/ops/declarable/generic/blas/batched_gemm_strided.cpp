//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/blas.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(batched_gemm_strided, 4, 1, false, 0, 9) {
            auto alphas = INPUT_VARIABLE(0);
            auto betas = INPUT_VARIABLE(1);
            auto A = INPUT_VARIABLE(2);
            auto B = INPUT_VARIABLE(3);
            auto C = OUTPUT_VARIABLE(4);

            return ND4J_STATUS_OK;
        };

        DECLARE_SHAPE_FN(batched_gemm_strided) {
            auto shapeList = new ShapeList();


            return shapeList;
        }
    }
}