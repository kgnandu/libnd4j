//
// Created by yurii@skymind.io on 02.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(reverse, 1, 1, true, 0, -2) {
	
	NDArray<T>* input  = INPUT_VARIABLE(0);
	NDArray<T>* output = OUTPUT_VARIABLE(0);

	std::vector<int>* argI = block.getIArguments();
	std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), *argI);

	ArrayList<T>* listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);
	ArrayList<T>* listIn  = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);
		
	NDArray<T>* subArrIn  = nullptr;
	NDArray<T>* subArrOut = nullptr;
	int subArrLength = 0;
	for(int i=0; i<listIn->size(); ++i) {		// listIn->size() = listOut->size()
		subArrIn   = listIn->at(i);
		subArrOut  = listOut->at(i);
		subArrLength = subArrIn->lengthOf();
		for(int j=0; j<subArrLength; ++j)
			subArrOut->putIndexedScalar(j, subArrIn->getIndexedScalar(subArrLength - 1 - j));
	}

	output->printBuffer("output");

	STORE_RESULT(*output);

	delete listOut;
	delete listIn;

	return ND4J_STATUS_OK;
}




}
}