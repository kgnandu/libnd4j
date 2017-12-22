//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/batched_gemm_strided.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(batched_gemm_strided, 4, 1, false, 0, 0) {
            auto alphas = INPUT_VARIABLE(0);
            auto betas = INPUT_VARIABLE(1);
            auto A = INPUT_VARIABLE(2);
            auto B = INPUT_VARIABLE(3);
            auto C = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(A->rankOf() == 3 && B->rankOf() == 3 && C->rankOf() == 3, 0, "BatchedGemmStrided: A, B and C should be rank 3 arrays");
            REQUIRE_TRUE(A->sizeAt(0) == B->sizeAt(0) && A->sizeAt(0) == C->sizeAt(0), 0, "BatchedGemmStrided: number of subarrays in batch should match for all A, B, C");
            int batchSize = A->sizeAt(0);

            REQUIRE_TRUE(batchSize == alphas->lengthOf() && batchSize == betas->lengthOf(), 0, "BatchedGemmStrided: lengths of Alpha and Beta should match batch size of %i, but got alpha of %i, and beta of % instead", batchSize, alphas->lengthOf(), betas->lengthOf());

            nd4j::ops::helpers::_bgemms(A, B, C, alphas, betas);

            return ND4J_STATUS_OK;
        };

        DECLARE_SHAPE_FN(batched_gemm_strided) {
            auto shapeList = new ShapeList();            

            auto A = inputShape->at(2);
            auto B = inputShape->at(3);

            int batchSize = shape::sizeAt(A, 0);
            int M = shape::sizeAt(A, 1);
            int N = shape::sizeAt(B, 2);

            std::vector<int> shape({batchSize, M, N});

            for (int e = 0; e < batchSize; e++) {
                int *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(3), int);

                shape::shapeBuffer(3, shape.data(), newShape);

                shapeList->push_back(newShape);
            }

            return shapeList;
        }
    }
}