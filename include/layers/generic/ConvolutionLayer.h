//
// @author raver119@gmail.com
//

#ifndef PROJECT_CONVOLUTIONLAYER_H
#define PROJECT_CONVOLUTIONLAYER_H

#include <layers/layers.h>
#include <layers/generic/BaseLayer.h>

namespace nd4j {
namespace layers {

template<typename T, typename AF> class ConvolutionLayer: public BaseLayer<T, AF> {
    private:
        int _kernelH, _kernelW;             // kernel sizes
        int _strideH, _strideW;             // step of kernel slide across the width and height of the input volume/picture 
        int _padH, _padW;                   // the number of zero-columns and zero-rows at the edges of input volume/picture 
        bool _padModeSame;

    public:
        // default constructor 
        ConvolutionLayer() = delete;

        // constructor 
        ConvolutionLayer(const int kernelH, const int kernelW, const int strideH, const int strideW, const int padH, const int padW, const bool padModeSame);
        
        // This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
        virtual int validateInput() const;

        // This method should validate output parameters, and return TRUE if everything is ok, FALSE otherwise        
        virtual int validateOutput() const;

        // This method "allocates" memory chunk from workspace
        // virtual T* allocate(long bytes) = 0; 
        
        // This method should validate parameters & bias, and return TRUE if everything ok. False otherwise
        virtual int validateParameters() const;

        // this method should validate memory/holders for BP pass
        virtual int validateGradients() const;

        // feed forward
        virtual int feedForward();

        // back propagate
        virtual int backPropagate();
        
};



//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

// constructor 
template<typename T, typename AF> ConvolutionLayer<T,AF>::ConvolutionLayer(const int kernelH, const int kernelW, const int strideH, const int strideW, const int padH, const int padW, const bool padModeSame): BaseLayer<T,AF>()  {
    _kernelH     = kernelH;  
    _kernelW     = kernelW;
    _strideH     = strideH;
    _strideW     = strideW;    
    _padH        = padH;     
    _padW        = padW; 
    _padModeSame = padModeSame;
}

//////////////////////////////////////////////////////////////////////
// This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF> int ConvolutionLayer<T,AF>::validateInput() const {
    
   if (this->_input == nullptr || this->_input->getShapeInfo() == nullptr || this->_input->getBuff() == nullptr)        
        return ND4J_STATUS_BAD_INPUT;

    if (this->_params == nullptr || this->_params->getShapeInfo() == nullptr || this->_params->getBuff() == nullptr)
        ND4J_STATUS_BAD_PARAMS;

    if (this->_bias == nullptr || this->_bias->getShapeInfo() == nullptr || this->_bias->getBuff() == nullptr)
        ND4J_STATUS_BAD_BIAS;

    if (this->_input->rankOf() != 4 || this->_params->rankOf() != 4 || this->_bias->rankOf() != 2)
        return ND4J_STATUS_BAD_RANK;

    if (this->_input->getShapeInfo()[2] != this->_params->getShapeInfo()[2])
        return ND4J_STATUS_BAD_SHAPE;        
    
}

//////////////////////////////////////////////////////////////////////
// This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
template<typename T, typename AF> int ConvolutionLayer<T,AF>::validateOutput() const {
    
    if ((this->_output == nullptr || !this->_output->nonNull()))
        return ND4J_STATUS_BAD_OUTPUT;

    // if (this->_output->rankOf() != 6)
        // return ND4J_STATUS_BAD_RANK;

        
    return ND4J_STATUS_OK;
}

//////////////////////////////////////////////////////////////////////
template<typename T, typename AF> int ConvolutionLayer<T,AF>::validateParameters() const {
    
    if (this->_params == nullptr || this->_params->getShapeInfo() == nullptr || this->_bias->getShapeInfo() == nullptr || 
        this->_bias == nullptr || this->_params->getBuff() == nullptr || this->_bias->getBuff() == nullptr )
        return ND4J_STATUS_BAD_PARAMS;    
    
    if (this->_params->rankOf() != 4 || this->_bias->rankOf() != 2)
        return ND4J_STATUS_BAD_RANK;

    // check the weight matrix consistency with class kernel members 
    if (this->_params->getShapeInfo()[3] != _kernelH || this->_params->getShapeInfo()[4] != _kernelW)
        return ND4J_STATUS_BAD_PARAMS;

    if (this->_params->getShapeInfo()[1] != this->_bias->getShapeInfo()[2])
        ND4J_STATUS_BAD_SHAPE;
    
    return ND4J_STATUS_OK;
}


//////////////////////////////////////////////////////////////////////
template<typename T, typename AF> int ConvolutionLayer<T,AF>::validateGradients() const {

}

//////////////////////////////////////////////////////////////////////
// feed forward
template<typename T, typename AF> int ConvolutionLayer<T,AF>::feedForward() {
   
    // // gemm here, input * W
    // // these values should be set appropriately

    // this->gemmHelper(this->_input, this->_params, this->_output, (T) 1.0f, (T) 0.0f);

    // // we're rolling through rows here
    // this->_output->addiRowVector(this->_bias);
   

    // // activation call
    // ActivationsExecutioner<T>::template executeFF<AF>(this->_output, this->_output);

    // return ND4J_STATUS_OK;
}

// back propagation
template<typename T, typename AF> int ConvolutionLayer<T,AF>::backPropagate() {

} 

// end of namespace brackets
}
}


#endif //PROJECT_CONVOLUTIONLAYER_H
