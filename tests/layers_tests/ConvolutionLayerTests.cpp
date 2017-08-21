//
// Created by shyrma on 21.08.2017
//

#include "testlayers.h"
#include <layers/generic/ConvolutionLayer.h>

class ConvolutionLayerTest : public testing::Test {
public:
    static const int bS = 20;       // batch size
    static const int iD = 3;        // input depth (number of picture channels, for example rgb=3)
    static const int pH = 256;      // picture height in pixels 
    static const int pW = 256;      // picture width in pixels 
    static const int oD = 5;        // output depth (= N for dense layer)
    static const int kH = 3;        // kernel height in pixels 
    static const int kW = 3;        // kernel width in pixels 

    int shapeI[12]  = {4, bS, iD, pH, pW, iD*pH*pW, pH*pW, pW, 1, 0, 1, 99}; 
    int shapeW[12]  = {4, oD, iD, kH, kW, iD*kH*kW, kH*kW, kW, 1, 0, 1, 99}; 
    int shapeB[8]   = {2, 1, oD, oD, 1, 0, 1, 99};
    int shapeO[12]  = {4, bS, iD, pH, pW, iD*pH*pW, pH*pW, pW, 1, 0, 1, 99}; 
};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

TEST_F(ConvolutionLayerTest, ValidationTest) {

    float* input = new float[bS*iD*pH*pW];
    float* output = new float[bS*iD*pH*pW];    
    nd4j::layers::ConvolutionLayer<float, nd4j::activations::Identity<float>> layer(3, 3, 1, 1, 0, 0, true);
    layer.getParams()->setShape(shapeW);
    layer.getBias()->setShape(shapeB);     
    
    int result = layer.validateParameters();
    ASSERT_EQ(result, ND4J_STATUS_OK);         
    
    result = layer.configureLayerFF(input, shapeI, output, shapeO, 0.f, 0.f, nullptr);
    ASSERT_EQ(result, ND4J_STATUS_OK);
    
    delete []input;
    delete []output;
}


