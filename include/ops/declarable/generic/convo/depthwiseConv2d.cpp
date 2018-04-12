//
// created by Yurii Shyrma on 08.03.2018
//


#include <op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <declarable/generic/helpers/convolutions.h>


namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(depthwise_conv2d, 2, 1, false, 0, 9) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, mC] (NHWC) or [mC, iC, kH, kW] (NCHW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC] = iC*mC
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)        
                                     
    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 0-NCHW,  1-NHWC

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weights->sizeAt(indWmC);                           // channels multiplier
    
    std::vector<std::vector<int>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
    std::vector<std::vector<int>> modifWeights, modifOutput;
    std::vector<int> outReShape;

    if(!isNCHW) {        
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        outReShape = {bS, oH, oW, iC, mC};                                              // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifOutput = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifWeights = {{2,0,1,3},{iC,kH*kW,mC}};                                       // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
    }
    else {
        outReShape = {bS, iC, mC, oH, oW};                                              // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifOutput = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifWeights = {{1,2,3,0},{iC,kH*kW,mC}};                                       // [mC,iC,kH,kW]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]           
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, block.getWorkspace());                
    NDArray<T>* outputReshaped = output->reshape(output->ordering(), outReShape);
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});

    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                                 // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    nd4j::NDArrayFactory<T>::tensorDot(&columns, weights, outputReshaped, modifColumns, modifWeights, modifOutput);    // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]
    
    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({indIOioC}, bias);

    if(!isNCHW)
        delete input;                  
    
    delete outputReshaped;

    return Status::OK();
}



DECLARE_SHAPE_FN(depthwise_conv2d) {

    int* inputShapeInfo   = inputShape->at(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    int* weightsShapeInfo = inputShape->at(1);                                    // [kH, kW, iC, mC] (NHWC) or [mC, iC, kH, kW] (NCHW)
    int* biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;      // [oC] = iC*mC

    const int rank = 4;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM DEPTHWISECONV2D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM DEPTHWISECONV2D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);

    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NDHWC, 1-NCDHW

    int indIOioC, indIiH, indWkH, indWmC, indWiC;
    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indWmC = 3; indWiC = 2;
    }
    else {        
        indIOioC = 1; indIiH = 2; indWkH = 2; indWmC = 0; indWiC = 1;              
    }    

    const int bS = inputShapeInfo[1];                            // batch size
    const int iH = inputShapeInfo[indIiH+1];                     // input height
    const int iW = inputShapeInfo[indIiH+2];                     // input width
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels        
    const int mC = weightsShapeInfo[indWmC+1];                   // channels multiplier(oC = iC*mC)
    const int oC = iC*mC;                                        // output channels

    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({iC,mC,kH,kW,  indWiC,indWmC,indWkH,indWkH+1}));        
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weightsShapeInfo), 0, "DEPTHWISECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weightsShapeInfo).c_str());    
    if (biasShapeInfo) 
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "DEPTHWISECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));    

    int oH, oW;                                         // output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

    outputShapeInfo[0] = rank;
    outputShapeInfo[1] = bS;

    if (isNCHW) {
        outputShapeInfo[2] = oC;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
    } else {
        outputShapeInfo[2] = oH;
        outputShapeInfo[3] = oW;
        outputShapeInfo[4] = oC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}



////////////////////////////////////////////////////////////////////////// 
CUSTOM_OP_IMPL(depthwise_conv2d_bp, 3, 2, false, 0, 9) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC] = [iC*mC]
    NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]        
                                     
    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NHWC, 1-NCHW    

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weights->sizeAt(indWmC);                           // channels multiplier    

    std::vector<std::vector<int>> modifColumns = {{1,2,3,0,4,5}, {iC, kH*kW, bS*oH*oW}};      // [bS,iC,kH,kW,oH,oW] -> [iC, kH*kW, bS*oH*oW]
    std::vector<std::vector<int>> modifGradW, modifGradO1, modifGradO2;
    std::vector<int> gradOreShape;

    if(!isNCHW) {        
        input = input->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        gradI = gradI->permute({0, 3, 1, 2});                                           // [bS,iH,iW,iC]    -> [bS,iC,iH,iW] 
        gradOreShape = {bS, oH, oW, iC, mC};                                            // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
        modifGradO1 = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{3,0,1,2},{iC, mC, bS*oH*oW}};                                   // [bS,oH,oW,iC*mC] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        modifGradW = {{2,0,1,3},{iC,kH*kW,mC}};                                         // [kH,kW,iC,mC]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]
    }
    else {
        gradOreShape = {bS, iC, mC, oH, oW};                                            // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
        modifGradO1 = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
        modifGradO2 = {{1,0,2,3},{iC, mC, bS*oH*oW}};                                   // [bS,iC*mC,oH,oW] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
        modifGradW = {{1,2,3,0},{iC,kH*kW,mC}};                                         // [mC,iC,kH,kW]    -> [iC,kH,kW,mC]    -> [iC,kH*kW,mC]           
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T>  columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, block.getWorkspace());        
    NDArray<T>* gradOreshaped = gradO->reshape(gradO->ordering(), gradOreShape);
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    
    // ----- calculation of gradW and gradB ----- //            
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    nd4j::NDArrayFactory<T>::tensorDot(&columns, gradOreshaped, gradW, modifColumns, modifGradO1, modifGradW);  // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

    // ----- calculation of gradB ----- //
    if(gradB) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0,indOoH,indOoH+1});                      // sum over bS, oH, oW
        if(gradB != OUTPUT_VARIABLE(2)) 
            delete gradB;
    }

    //----- calculation of gradI -----//                
    nd4j::NDArrayFactory<T>::tensorDot(weights, gradO, &columns, modifGradW, modifGradO2, modifColumns); // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]    
    columns.template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());                     // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    if(!isNCHW) {        
        delete input;        
        delete gradI;
    }

    delete gradOreshaped;              

    return Status::OK();
}



DECLARE_SHAPE_FN(depthwise_conv2d_bp) {

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);
    int* biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;  
    int* gradOShapeInfo   = block.width() > 3 ? inputShape->at(3) : inputShape->at(2);  

    const int rank = 4;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);
    REQUIRE_TRUE(gradOShapeInfo[0]   == rank, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of gradO array must be equal to %i, but got %i instead !", rank, gradOShapeInfo[0]);

    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NHWC, 1-NCHW    

    int indIOioC, indIiH, indWkH, indWmC, indWiC, indOoH;
    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indWmC = 3; indWiC = 2; 
    }
    else {        
        indIOioC = 1; indIiH = 2; indWkH = 2; indWmC = 0; indWiC = 1;              
    }    

    const int bS = inputShapeInfo[1];                            // batch size
    const int iH = inputShapeInfo[indIiH+1];                     // input height
    const int iW = inputShapeInfo[indIiH+2];                     // input width
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels        
    const int mC = weightsShapeInfo[indWmC+1];                   // channels multiplier(oC = iC*mC)
    const int oC = iC*mC;                                        // output channels

    int trueoH, trueoW;          // correct output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indIiH,indIiH+1}));            
    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({mC,iC,kH,kW,  indWmC,indWiC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils<T>::shapeAsString(gradOShapeInfo), 0,  "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of gradient_output (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils<T>::shapeAsString(gradOShapeInfo).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weightsShapeInfo), 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weightsShapeInfo).c_str());    
    if(biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));        

    int* gradIshapeInfo(nullptr), *gradWshapeInfo(nullptr);
    COPY_SHAPE(inputShapeInfo, gradIshapeInfo);
    COPY_SHAPE(weightsShapeInfo, gradWshapeInfo);

    if(biasShapeInfo) {
        int* gradBshapeInfo(nullptr);
        COPY_SHAPE(biasShapeInfo, gradBshapeInfo);
        return SHAPELIST(gradIshapeInfo, gradWshapeInfo, gradBshapeInfo);
    }     

    return SHAPELIST(gradIshapeInfo, gradWshapeInfo);        
}




}
}