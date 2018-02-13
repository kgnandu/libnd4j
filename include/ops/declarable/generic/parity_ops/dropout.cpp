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

    Nd4jIndex *buffer = new Nd4jIndex[100000];
    NativeOps nativeOps;
    //int seed = 100;
    nd4j::random::RandomBuffer *rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, seed, 100000, (Nd4jPointer) buffer);

    if (rng == nullptr)
        throw "RNG initialization failed";

    VariableSpace<T>* variableSpace = new VariableSpace<T>();
    std::unique_ptr<NDArray<T>> tempInput(new NDArray<T>(*input));
    variableSpace->putVariable(-1, tempInput.get());
    Context<T>* theVBlock = new Context<T>(1, variableSpace, true);
    theVBlock->fillInputs({-1});
    theVBlock->setRNG(rng);
    theVBlock->getTArguments()->push_back(probValue);
    theVBlock->getTArguments()->push_back(probValue + T(1.0f));

    nd4j::ops::randomuniform<T> uniform;

    Nd4jStatus  status = uniform.execute(theVBlock);

    REQUIRE_TRUE(ND4J_STATUS_OK == status, 0, "dropout: Cannot make uniform matrix for dropout process");

    tempInput->template applyTransform<simdOps::Floor<T>>();
    tempInput->template applyScalar<simdOps::Multiply<T>>(T(1.0) / probValue);

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;

    delete variableSpace;
    delete theVBlock;

    *output = *input * *tempInput;

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