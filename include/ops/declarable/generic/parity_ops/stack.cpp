//
// Created by yurii@skymind.io on 01.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <vector>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(stack, -1, 1, false, 0, 1) {

    int dim = block.getIArguments()->at(0);
    if(dim < 0)
    	dim += (INPUT_VARIABLE(0))->rankOf();                

    NDArray<T> *output = OUTPUT_VARIABLE(0);

    int inArrNum = (int)block.width();
    Nd4jPointer* buffers = new Nd4jPointer[inArrNum];
    Nd4jPointer* shapes = new Nd4jPointer[inArrNum];

    Variable<T>* var = nullptr;
    for (int e = 0; e < inArrNum; e++) {
        var = block.variable(e);
        buffers[e] = (Nd4jPointer) var->getNDArray()->getBuffer();
        shapes[e] = (Nd4jPointer) var->getNDArray()->getShapeInfo();
    }

    nd4j::SpecialMethods<T>::concatCpuGeneric(dim, inArrNum, buffers, shapes, output->getBuffer(), output->getShapeInfo());

    STORE_RESULT(*output);

    delete[] buffers;
    delete[] shapes;

    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(stack) {

	// check whether shapes of all input array are the same	
	const int inArrNum = (int) block.width();
	for (int i = 0; i < inArrNum - 1; ++i)
		if (!shape::equalsSoft(inputShape->at(i), inputShape->at(i+1)))
			throw "CUSTOM_OP stack: the shapes of input arrays are different !";
	// check whether input dimension is within rank range
	int* inShapeInfo = inputShape->at(0);
	int rank = inShapeInfo[0];
	int dim = block.getIArguments()->at(0);
	if(dim < 0 ) dim += rank;
	if(dim >= rank)
		throw "CUSTOM_OP stack: the input dimension is greater/equal than rank of input input arrays shapes !";

	//the rank of output ShapeInfo is larger by one compared to input ShapeInfo
	std::vector<int> outShape(inShapeInfo + 1, inShapeInfo + 1 + rank);
	// insert inArrNum at dim position of input shape to get output shape	
	outShape.insert(outShape.begin() + dim, inArrNum);
	// if input arrays are vectors remove unity from shape
	NDArray<T>* input = INPUT_VARIABLE(0);
	if(input->isVector())
		outShape.erase(std::remove(outShape.begin(), outShape.end(), 1), outShape.end());

	// evaluate output ShapeInfo
	int newRank = outShape.size();
	int* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), newRank*2+4, int);
    outShapeInfo[0] = newRank;
    for(int i=1; i <= newRank; ++i)
    	outShapeInfo[i] = outShape[i-1];
	
    shape::updateStrides(outShapeInfo, input->ordering());

    return new ShapeList(outShapeInfo);
    

}



















}
}