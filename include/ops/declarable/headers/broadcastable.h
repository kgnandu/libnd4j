//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_BROADCASTABLE_H
#define LIBND4J_HEADERS_BROADCASTABLE_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        // TODO: make broadcastables separate class

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Max(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__maximum)
        DECLARE_CUSTOM_OP(maximum, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(maximum_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Min(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__minimum)
        DECLARE_CUSTOM_OP(minimum, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(minimum_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Add(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__add)
        DECLARE_CUSTOM_OP(add, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(add_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Subtract(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__subtract)
        DECLARE_CUSTOM_OP(subtract, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(subtract_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Subtract(Y, X)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__reversesubtract)
        DECLARE_CUSTOM_OP(reversesubtract, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(reversesubtract_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = ReverseMod(X, Y) == Mod(Y, X)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__reversemod)
        DECLARE_CUSTOM_OP(reversemod, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(reversemod_bp, 3, 2, true, 0, 0);
        #endif
        

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Subtract(X, Y) * Subtract(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__squaredsubtract)
        DECLARE_CUSTOM_OP(squaredsubtract, 2, 1, true, 0, 0)
        DECLARE_CUSTOM_OP(squaredsubtract_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Multiply(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__multiply)
        DECLARE_CUSTOM_OP(multiply, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(multiply_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Divide(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__divide)
        DECLARE_CUSTOM_OP(divide, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(divide_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Divide(Y, x)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__reversedivide)
        DECLARE_CUSTOM_OP(reversedivide, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(reversedivide_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = FloorMod(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__floormod)
        DECLARE_CUSTOM_OP(floormod, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(floormod_bp, 3, 2, true, 0, 0);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__mod)
        DECLARE_CUSTOM_OP(mod, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(mod_bp, 3, 2, true, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = FloorDiv(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__floordiv)
        DECLARE_CUSTOM_OP(floordiv, 2, 1, true, 0, 0)
        DECLARE_CUSTOM_OP(floordiv_bp, 2, 1, true, 0, 0)
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Divide(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__realdiv)
        DECLARE_CUSTOM_OP(realdiv, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(realdiv_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * This is one of auto-broadcastable operations. It accepts 2 operands, and operation is applied based on their shapes:
         * 1) if shapes are equal that's pairwise operation, result will have the same shape.
         * 2) if shape X is scalar and shape Y is array - result will have shape equal to Y.
         * 3) if shape X is array and shape Y is scalar - result will have shape equal to X.
         * 4) if shape X and Y are both arrays, but shapes aren't equal - result shape will be broadcast result.
         * 
         * This operation returns Z = Assign(X, Y)
         */
        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__assign)
        DECLARE_CUSTOM_OP(assign, 2, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(assign_bp, 3, 2, false, 0, 0);
        #endif

        #if defined(__ALL_OPS) || defined(__CLION_IDE__) || defined(__meshgrid)
        DECLARE_CUSTOM_OP(meshgrid, -1, -1, false, 0, 0);
        #endif
    }
}

#endif