//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_LIST_H
#define LIBND4J_HEADERS_LIST_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        // list operations, basically all around NDArrayList

        /**
         * This operations puts given NDArray into (optionally) given NDArrayList. 
         * If no NDArrayList was provided - new one will be created
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__write_list)
        DECLARE_LIST_OP(write_list, 2, 1, 0, -2);
        #endif

        /**
         * This operation concatenates given NDArrayList, and returns NDArray as result
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__stack_list)
        DECLARE_LIST_OP(stack_list, 1, 1, 0, 0);
        #endif

        /**
         * This operations selects specified index fron NDArrayList and returns it as NDArray
         * Expected arguments:
         * x: non-empty list
         * indices: optional, scalar with index
         * 
         * Int args:
         * optional, index
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__read_list)
        DECLARE_LIST_OP(read_list, 1, 1, 0, 0);
        #endif

        /**
         * This operations selects specified indices fron NDArrayList and returns them as NDArray
         * Expected arguments:
         * x: non-empty list
         * indices: optional, vector with indices
         * 
         * Int args:
         * optional, indices
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__pick_list)
        DECLARE_LIST_OP(pick_list, 1, 1, -2, -2);
        #endif

        /**
         * This operations returns scalar, with number of existing arrays within given NDArrayList
         * Expected arguments:
         * x: list
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__size_list)
        DECLARE_LIST_OP(size_list, 1, 1, 0, 0);
        #endif

        /**
         * This operation creates new empty NDArrayList
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__create_list)
        DECLARE_LIST_OP(create_list, 1, 2, 0, -2);
        #endif

        /**
         * This operation unpacks given NDArray into specified NDArrayList wrt specified indices
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__scatter_list)
        DECLARE_LIST_OP(scatter_list, 1, 1, 0, -2);
        #endif

        /**
         * This operation splits given NDArray into chunks, and stores them into given NDArrayList wert sizes
         * Expected arguments:
         * list: optional, NDArrayList. if not available - new NDArrayList will be created
         * array: array to be split
         * sizes: vector with sizes for each chunk
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__split_list)
        DECLARE_LIST_OP(split_list, 2, 1, 0, -2);
        #endif

        /**
         * This operation builds NDArray from NDArrayList using indices
         * Expected arguments:
         * x: non-empty list
         * indices: vector with indices for gather operation
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__gather_list)
        DECLARE_LIST_OP(gather_list, 2, 1, 0, -2);
        #endif

        /**
         * This operation clones given NDArrayList
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__clone_list)
        DECLARE_LIST_OP(clone_list, 1, 1, 0, 0);
        #endif

        /**
         * This operation unstacks given NDArray into NDArrayList
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__unstack_list)
        DECLARE_LIST_OP(unstack_list, 1, 1, 0, 0);
        #endif
    }
}

#endif