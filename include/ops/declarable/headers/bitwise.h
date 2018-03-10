//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_BITWISE_H
#define LIBND4J_HEADERS_BITWISE_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation toggles individual bits of each element in array
         * 
         * PLEASE NOTE: This operation is possible only on integer datatypes
         * 
         * @tparam T
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__toggle_bits)
        DECLARE_OP(toggle_bits, -1, -1, true);
        #endif
    }
}

#endif