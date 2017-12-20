//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/blas.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(batched_gemm_strided, 4, 1, false, 0, 12) {
            auto alphas = INPUT_VARIABLE(0);
            auto betas = INPUT_VARIABLE(1);
            auto A = INPUT_VARIABLE(2);
            auto B = INPUT_VARIABLE(3);
            auto C = OUTPUT_VARIABLE(4);

            int transA = INT_ARG(0);
            int transB = INT_ARG(1);
            int M = INT_ARG(2);
            int N = INT_ARG(3);
            int K = INT_ARG(4);
            int ldA = INT_ARG(5);
            int ldB = INT_ARG(6);
            int ldC = INT_ARG(7);
            int strideA = INT_ARG(8);
            int strideB = INT_ARG(9);
            int strideC = INT_ARG(10);
            int batchSize = INT_ARG(11);

            if (transA == 0)
                transA = 111;
            
            if (transB == 0)
                transB = 111;

            if (transA == 1)
                transA = 112;
            
            if (transB == 1)
                transB = 112;

            REQUIRE_TRUE((transA == 111 || transA == 112) && (transB == 111 || transB == 112), 0, "BatchedGemmStrided: valid values for transA and transB are: 0/1 or 111/112, for NoTrans/Trans respectively")
            REQUIRE_TRUE(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0, 0, "BatchedGemmStrided: M, N, K, ldA, ldB, ldC and batchSize should have positive values");

            return ND4J_STATUS_OK;
        };

        DECLARE_SHAPE_FN(batched_gemm_strided) {
            auto shapeList = new ShapeList();

            int transA = INT_ARG(0);
            int transB = INT_ARG(1);
            int M = INT_ARG(2);
            int N = INT_ARG(3);
            int K = INT_ARG(4);
            int ldA = INT_ARG(5);
            int ldB = INT_ARG(6);
            int ldC = INT_ARG(7);
            int strideA = INT_ARG(8);
            int strideB = INT_ARG(9);
            int strideC = INT_ARG(10);
            int batchSize = INT_ARG(11);

            if (!(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0)) {
                int *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);

                newShape[0] = 2;
                newShape[1] = 1;
                newShape[2] = 1;
                newShape[3] = 1;
                newShape[4] = 1;
                newShape[5] = 0;
                newShape[6] = 1;
                newShape[7] = 99;

                shapeList->push_back(newShape);
                return shapeList;
            }
            

            std::vector<int> shape({batchSize, M, N});

            for (int e = 0; e < batchSize; e++) {
                int *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(3), int);

                shape::shapeBufferFortran(3, shape.data(), newShape);

                shapeList->push_back(newShape);
            }

            return shapeList;
        }
    }
}