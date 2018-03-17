//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_TRANSFORMS_H
#define LIBND4J_HEADERS_TRANSFORMS_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__clipbyvalue)
        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__clipbynorm)
        DECLARE_CONFIGURABLE_OP(clipbynorm, 1, 1, true, 1, 0);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__clipbyavgnorm)
        DECLARE_CONFIGURABLE_OP(clipbyavgnorm, 1, 1, true, 1, 0);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__cumsum)
        DECLARE_CONFIGURABLE_OP(cumsum, 1, 1, true, 0, -2);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__cumprod)
        DECLARE_CONFIGURABLE_OP(cumprod, 1, 1, true, 0, -2);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__tile)
        DECLARE_CUSTOM_OP(tile, 1, 1, false, 0, -2);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__repeat)
        DECLARE_CUSTOM_OP(repeat, 1, 1, true, 0, -1); 
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__invert_permutation)
        DECLARE_CONFIGURABLE_OP(invert_permutation, 1, 1, false, 0, 0);  
        #endif

        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(concat_bp, -1, -1, false, 0, 1);

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__mergemax)
        DECLARE_OP(mergemax, -1, 1, false);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__mergemaxindex)
        DECLARE_OP(mergemaxindex, -1, 1, false);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__mergeadd)
        DECLARE_OP(mergeadd, -1, 1, false);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__mergeavg)
        DECLARE_OP(mergeavg, -1, 1, false);   
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__scatter_update)
        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1); 
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__Floor)
        DECLARE_OP(Floor, 1, 1, true);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__Log1p)
        DECLARE_OP(Log1p, 2, 1, true);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__reverse)
        DECLARE_CONFIGURABLE_OP(reverse, 1, 1, true, 0, -2);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__gather)
        DECLARE_CUSTOM_OP(gather, 1, 1, false, 0, 1);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__pad)
        DECLARE_CUSTOM_OP(pad, 2, 1, false, 0, 1);
        #endif

        /**
         * creates identity 2D matrix or batch of identical 2D identity matrices
         * 
         * Input array:
         * provide some array - in any case operation simply neglects it
         * 
         * Input integer arguments:
         * IArgs[0]       - order of output identity matrix, 99 -> 'c'-order, 102 -> 'f'-order
         * IArgs[1]       - the number of rows in output inner-most 2D identity matrix
         * IArgs[2]       - optional, the number of columns in output inner-most 2D identity matrix, if this argument is not provided then it is taken to be equal to number of rows
         * IArgs[3,4,...] - optional, shape of batch, output matrix will have leading batch dimensions of this shape         
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__eye)
        DECLARE_CUSTOM_OP(eye, 1, 1, false, 0, 2);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__gather_nd)
        DECLARE_CUSTOM_OP(gather_nd, 2, 1, false, 0, 0);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__reverse_sequence)
        DECLARE_CUSTOM_OP(reverse_sequence, 2, 1, false, 0, 2);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__trace)
        DECLARE_CUSTOM_OP(trace, 1, 1, false, 0, 0);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__random_shuffle)
        DECLARE_OP(random_shuffle, 1, 1, true);
        #endif

        /**
         * clip a list of given tensors with given average norm when needed
         * 
         * Input:
         *    a list of tensors (at least one)
         * 
         * Input floating point argument:
         *    clip_norm - a value that used as threshold value and norm to be used
         *
         * return a list of clipped tensors
         *  and global_norm as scalar tensor at the end
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__clip_by_global_norm)
        DECLARE_CUSTOM_OP(clip_by_global_norm, 1, 2, true, 1, 0);
        #endif
    }
}

#endif