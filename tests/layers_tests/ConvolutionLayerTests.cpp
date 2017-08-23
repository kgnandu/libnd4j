//
// Created by shyrma on 21.08.2017
//

#include "testlayers.h"
#include "confo_def.h"
#include <layers/generic/ConvolutionLayer.h>

class ConvolutionLayerTest : public testing::Test {
public:

};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

// TEST_F(ConvolutionLayerTest, ValidationTest) {

    // float* input  = new float[bS*iD*pH*pW];
    // float* output = new float[bS*oD*oH*oW];    
    
    // nd4j::layers::ConvolutionLayer<float, nd4j::activations::Identity<float>> layer(kH, kW, sW, sH, pdW, pdH, true);
    // layer.getParams()->setShape(shapeW);
    // layer.getBias()->setShape(shapeB);     
    
    // int result = layer.validateParameters();
    // ASSERT_EQ(result, ND4J_STATUS_OK);         
    
    // result = layer.configureLayerFF(input, shapeI, output, shapeZ, 0.f, 0.f, nullptr);
    // ASSERT_EQ(result, ND4J_STATUS_OK);
    
    // delete []input;
    // delete []output;
// }

TEST_F(ConvolutionLayerTest, FFtest) {
        
    nd4j::layers::ConvolutionLayer<float, nd4j::activations::Identity<float>> layer(kH, kW, sW, sH, pdW, pdH, true);    
    NDArray<float> finalMatrix(Z, shapeZ);
    
    float* ob = new float[bS*oD*oH*oW];
    NDArray<float> input(I, shapeI);
    NDArray<float> output(ob, shapeZ);
    int result = layer.setParameters(W, shapeW, B, shapeB);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    printf("Input shape: \n");
    input.printShapeInfo();
        
    result = layer.configureLayerFF(input.getBuff(), input.getShapeInfo(), output.getBuff(), output.getShapeInfo(), 0.f, 0.f, nullptr);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    result = layer.feedForward();                
    ASSERT_EQ(ND4J_STATUS_OK, result);


    printf("Output: \n");
    output.print();

    printf("\nExpected: \n");
    finalMatrix.print();

    //for(int i=0; i<layer.getOutput()->lengthOf(); ++i)
//        std::cout<<std::setw(10)<<layer.getOutput()->getBuff()[i]<<"  "<<std::setw(10)<<Z[i]<<std::endl;
    ASSERT_TRUE(finalMatrix == *layer.getOutput());        
    //delete []output;
}



