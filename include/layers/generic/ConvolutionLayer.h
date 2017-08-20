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
        inline virtual int validateInput() const;

        // This method should validate output parameters, and return TRUE if everything is ok, FALSE otherwise        
        inline virtual int validateOutput() const;

        // feed forward
        virtual int feedForward();

        // back propagate
        virtual int backPropagate();
        
};



//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

// default constructor 
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
// template<typename T, typename AF> virtual int ConvolutionLayer<T,AF>::backPropagate() {
// } 

// end of namespace brackets
}
}


#endif //PROJECT_CONVOLUTIONLAYER_H
