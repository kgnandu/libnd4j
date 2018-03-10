

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/compare_elem.h>

#if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__is_numeric_tensor)

namespace nd4j {
    namespace ops {
        BOOLEAN_OP_IMPL(is_numeric_tensor, 1, true) {

            auto input = INPUT_VARIABLE(0);

            return ND4J_STATUS_TRUE;
        }
    }
}

#endif