//
//  @author sgazeos@gmail.com
//  

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/top_k.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(top_k, 1, 2, false, 0, -2) {
            NDArray<T>* x = INPUT_VARIABLE(0);
            int k = 1;// from params
            bool needSort = true;
            NDArray<T>* values = OUTPUT_VARIABLE(0);
            NDArray<T>* indeces = OUTPUT_VARIABLE(1);
            if (block.numI() > 0) {
                k = INT_ARG(0);
                needSort = INT_ARG(1);
            }

            REQUIRE_TRUE(k <= x->sizeAt(-1), 0, "top_k: k should not be greater than last dimension");
            REQUIRE_TRUE(k >=0, 0, "top_k: k should be non-negative");

            return helpers::topKFunctor(x, values, indeces, k, needSort);
        }

        DECLARE_SHAPE_FN(top_k) {
            auto shapeList = SHAPELIST(); 
            auto in = inputShape->at(0);
            int shapeRank = shape::rank(in);
            int k = 1; // default output shape is size 1

            if (block.numI() > 0) {
                k = INT_ARG(0);
            }

            for (int e = 0; e < 2; e++) { // 2 element tuple at output
                int* newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(shapeRank), int);
                std::vector<int> internalShape(shapeRank);
                for (int e = 0 ; e < shapeRank - 1; ++e)
                    internalShape[e] = shape::sizeAt(in, e);
                internalShape[shapeRank - 1] = k;

                if (shape::order(in) == 'c')
                    shape::shapeBuffer(shapeRank, internalShape.data(),  newshape);
                else
                    shape::shapeBufferFortran(shapeRank, internalShape.data(),  newshape);

                shapeList->push_back(newshape); 
            }
            return shapeList;
        }
    }
}