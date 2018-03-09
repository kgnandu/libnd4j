//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_TPARTY_H
#define LIBND4J_HEADERS_TPARTY_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        DECLARE_CUSTOM_OP(firas_sparse, 1, 1, false, 0, -1);
        DECLARE_CUSTOM_OP(test_scalar, 1, 1, false, 0, 0);
    }
}

#endif