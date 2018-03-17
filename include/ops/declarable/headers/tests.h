//
//  @author raver119@gmail.com
//
#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__test_output_reshape)
        DECLARE_OP(test_output_reshape, 1, 1, true);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__test_scalar)
        DECLARE_CUSTOM_OP(test_scalar, 1, 1, false, 0, 0);
        #endif
    }
}