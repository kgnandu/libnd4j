//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/parity_ops.h>
//#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(in_top_k, 2, 1, true, 0, 1) {
            NDArray<T>* predictions = INPUT_VARIABLE(0);
            NDArray<T>* source = INPUT_VARIABLE(1);

            NDArray<T>* result = OUTPUT_VARIABLE(0);
            REQUIRE_TRUE(block.numI() > 0, 0, "Parameter k is needed to be set");

            int k = INT_ARG(0);

            //*** TO DO: remove this after stable working of top_k
            return ND4J_STATUS_BAD_ARGUMENTS;

            nd4j::ops::top_k<T> op;
            auto topKResult = op.execute({source}, {}, {k, 1}); // with sorting
            if (topKResult->status() != ND4J_STATUS_OK)
                return topKResult->status();

            for (int e = 0; e < predictions->lengthOf(); e++) {
                bool found = false;
                for (int j = 0; j < topKResult->at(0)->lengthOf(); j++) {
                    if (predictions->getScalar(e) == topKResult->at(0)->getScalar(j)) {
                        found = true;
                        break;
                    }
                }
                //if (prediction.getScalar(e) in topKResult.at(0)) { // the first result
                if (found)
                    result->putScalar(e, 1);
                else
                    result->putScalar(e, 0);
            }
            return ND4J_STATUS_OK;
        }

    }
}