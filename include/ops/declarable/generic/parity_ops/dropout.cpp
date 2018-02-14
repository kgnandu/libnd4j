//
// Created by GS <sgazeos@gmail.com>
//

#include <NativeOps.h>
#include <ops/declarable/CustomOperations.h>

//#include <helpers/ShapeUtils.h>
//#include <vector>
//#include <numeric>


namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(dropout, 3, 1, true, 0, 1) {
    NDArray<T>* input   = INPUT_VARIABLE(0); // lookup param
    NDArray<T>* probability = INPUT_VARIABLE(1); // indeces, as is
    NDArray<T>* reduceShape = INPUT_VARIABLE(2);
    NDArray<T>* output  = OUTPUT_VARIABLE(0); // 
    
    int seed = INT_ARG(0);

    REQUIRE_TRUE(probability->isScalar(), 0, "dropout: Need a scalar with range 0 to 1 as probability.");
    T probValue = probability->getScalar(0);

    REQUIRE_TRUE(probValue > T(0.f) && probValue <= T(1.f), 0, "dropout: Probability should be with range 0 to 1.");

    if (probValue == T(1.0)) {
        *output = *input;
        return ND4J_STATUS_OK;
    }

    REQUIRE_TRUE(reduceShape->lengthOf() <= input->rankOf(), 0, "dropout: Noise shape should be fittable to input");

    std::vector<Nd4jIndex> buffer(100000);
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, seed, buffer.size(), (Nd4jPointer) &buffer[0]);

    if (rng == nullptr)
        return ND4J_STATUS_BAD_RNG;


    std::vector<int> dims(reduceShape->lengthOf());

    bool fit = true;
/*
    for( int i = 0; i < dims.size(); i++ ) {
        dims[i] = (*reduceShape)(i);
        for (int e = 0; e < input->rankOf(); ++e)
            if (input->sizeAt(e) % dims[i]) {
                fit = false;
                break;
            }

        if(!fit) break;
    }
*/
    // check dims to fit input
    REQUIRE_TRUE(fit, 0, "dropout: Noise shape should fit to input rank.");

//    Nd4jIndex tensorCount = x.tensorsAlongDimension({1});
//    nd4j_printf("Total subarray are %i\n", tensorCount);
    
    input->template applyRandom<randomOps::DropOutInverted<T>>(rng, nullptr, output, &probValue);

    nativeOps.destroyRandom(rng);

    return ND4J_STATUS_OK;
}
/*
DECLARE_SHAPE_FN(embedding_lookup) {

    int* inShapeInfo = inputShape->at(0);
    int* indecesShapeInfo = inputShape->at(1);
    int inRank = shape::rank(inShapeInfo);

    int outRank = inRank; 

    int* outShapeInfo = nullptr;
    
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), int);
    std::vector<int> shapeInfo(outRank);

    shapeInfo[0] = indecesShapeInfo[1]; // vector - how many elements
    for (int e = 1; e < outRank; e++)
        shapeInfo[e] = shape::sizeAt(inShapeInfo, e);
    if (shape::order(inShapeInfo) == 'c')
        shape::shapeBuffer(outRank, shapeInfo.data(),  outShapeInfo);
    else
        shape::shapeBufferFortran(outRank, shapeInfo.data(),  outShapeInfo);

    return new ShapeList(outShapeInfo);    
}

*/


}
}