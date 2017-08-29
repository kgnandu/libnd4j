//
// Created by shyrma on 21.08.2017
//

#include "testlayers.h"
#include "confo_def.h"
#include <layers/generic/ConvolutionLayer.h>

class ConvolutionLayerTest : public testing::Test {

};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

// TEST_F(ConvolutionLayerTest, ValidationTest) {

    // double* input  = new double[bS*iD*pH*pW];
    // double* output = new double[bS*oD*oH*oW];    
    
    // nd4j::layers::ConvolutionLayer<double, nd4j::activations::Identity<double>> layer(kH, kW, sW, sH, pdW, pdH, true);
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
        
    nd4j::layers::ConvolutionLayer<double, nd4j::activations::Identity<double>> layer(kH, kW, sW, sH, pdW, pdH, true);    
    NDArray<double> finalMatrix(const_cast<double*>(Z), const_cast<int*>(shapeZ));
    
    double* ob = new double[bS*oD*oH*oW];
    NDArray<double> input(const_cast<double*>(I), const_cast<int*>(shapeI));
    NDArray<double> output(const_cast<double*>(ob), const_cast<int*>(shapeZ));
    int result = layer.setParameters(const_cast<double*>(W), const_cast<int*>(shapeW), const_cast<double*>(B), const_cast<int*>(shapeB));
    ASSERT_EQ(ND4J_STATUS_OK, result);

    result = layer.configureLayerFF(input.getBuff(), input.getShapeInfo(), output.getBuff(), output.getShapeInfo(), 0.f, 0.f, nullptr);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    result = layer.feedForward();                    
    ASSERT_EQ(ND4J_STATUS_OK, result);
    
    // int count1=layer.getOutput()->lengthOf();
    // int count2=0;
    // for(int i=0; i<layer.getOutput()->lengthOf(); ++i) {
        
        // double calculated = layer.getOutput()->getBuff()[i];
        // double actual = Z[i];
        // if (fabs(calculated - actual) > 0.00001) {
            // std::cout<<std::setw(10)<<calculated<<"  "<<std::setw(10)<<actual<<std::endl;
            // ++count2;
        // }
    // }
    // std::cout<<"!!!!! "<<count1<<"  "<<count2<<std::endl;        
    ASSERT_TRUE(finalMatrix == *layer.getOutput());        
    delete []ob;
}
