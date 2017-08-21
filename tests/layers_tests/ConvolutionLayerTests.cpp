//
// Created by shyrma on 21.08.2017
//

#include "testlayers.h"
#include <layers/generic/ConvolutionLayer.h>

class ConvolutionLayerTest : public testing::Test {
public:
    static const int bS  = 20;       // batch size
    static const int iD  = 3;        // input depth (number of picture channels, for example rgb=3)
    static const int pH  = 256;      // picture height in pixels 
    static const int pW  = 256;      // picture width in pixels 
    static const int oD  = 5;        // output depth (= N for dense layer)
    static const int kH  = 4;        // kernel height in pixels 
    static const int kW  = 4;        // kernel width in pixels 
    static const int sH  = 1;        // stride step in horizontal direction
    static const int sW  = 1;        // stride step in vertical direction
    static const int pdH = 0;        // padding height
    static const int pdW = 0;        // padding width
    static const int oW  = (pW - kW + 2*pdW)/(sW+1); // output width
    static const int oH  = (pH - kH + 2*pdH)/(sH+1); // output width

    int shapeI[12]  = {4, bS, iD, pH, pW, iD*pH*pW, pH*pW, pW, 1, 0, 1, 99}; 
    int shapeW[12]  = {4, oD, iD, kH, kW, iD*kH*kW, kH*kW, kW, 1, 0, 1, 99}; 
    int shapeO[12]  = {4, bS, oD, oH, oW, oD*oH*oW, oH*oW, oW, 1, 0, 1, 99}; 
    int shapeB[8]   = {2, 1, oD, oD, 1, 0, 1, 99};    
};


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

TEST_F(ConvolutionLayerTest, ValidationTest) {

    float* input  = new float[bS*iD*pH*pW];
    float* output = new float[bS*oD*oH*oW];    
    
    nd4j::layers::ConvolutionLayer<float, nd4j::activations::Identity<float>> layer(kH, kW, sW, sH, pdW, pdH, true);
    layer.getParams()->setShape(shapeW);
    layer.getBias()->setShape(shapeB);     
    
    int result = layer.validateParameters();
    ASSERT_EQ(result, ND4J_STATUS_OK);         
    
    result = layer.configureLayerFF(input, shapeI, output, shapeO, 0.f, 0.f, nullptr);
    ASSERT_EQ(result, ND4J_STATUS_OK);
    
    delete []input;
    delete []output;
}


