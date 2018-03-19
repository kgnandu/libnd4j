//
//
//

#if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__noop)

#include <ops/declarable/headers/tests.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(noop, -1, -1, true) {
            // Fastest op ever.
            return ND4J_STATUS_OK;
        }
    }
}

#endif
