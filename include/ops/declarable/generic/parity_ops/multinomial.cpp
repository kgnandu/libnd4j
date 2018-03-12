//
// Created by GS <sgazeos@gmail.com>
//


#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>

#include <ops/declarable/helpers/multinomial.h>

namespace nd4j {
namespace ops {


    //////////////////////////////////////////////////////////////////////////
    CUSTOM_OP_IMPL(multinomial, 1, 1, true, 0, 1) {
        NDArray<T>* input   = INPUT_VARIABLE(0); // lookup param

        NDArray<T>* reduceShape = nullptr; // this param is optional
        NDArray<T>* output  = OUTPUT_VARIABLE(0); // 
    
        REQUIRE_TRUE(input->rankOf() == 2, 0, "multinomial: input should be a 2D tensor, but %i was given.", input->rankOf());

        int seed = 0; // default seed is used

        int numOfSamples = INT_ARG(0);
    
        if (block.getIArguments()->size() > 1)
            seed = INT_ARG(1);

        nd4j::random::RandomBuffer* rng = block.getRNG();
    
        if (rng == nullptr)
            return ND4J_STATUS_BAD_RNG;

        return helpers::multiNomialFunctor(rng, input, output, numOfSamples, seed);
    }

    DECLARE_SHAPE_FN(multinomial) {
        auto in = inputShape->at(0);
        int shapeRank = shape::rank(in);
        int lastDim = INT_ARG(0);
        int* newshape;

        ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(shapeRank), int);
        std::vector<int> data = {shape::sizeAt(in, 0), lastDim};

        if (shape::order(in) == 'c')
            shape::shapeBuffer(shapeRank, data.data(), newshape);
        else 
            shape::shapeBufferFortran(shapeRank, data.data(), newshape);

        return SHAPELIST(newshape);
    }

}
}