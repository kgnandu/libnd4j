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

        T *_extraParams;                    // extraparams for im2col

    public:
        // default constructor 
        ConvolutionLayer() = delete;

        // copy constructor
        // creation of this class objects by copying is not expected, therefore disable copy constructor 
        ConvolutionLayer(const ConvolutionLayer& ) = delete;
        
        // assignment operator
        // the assignment operations are not expected for this class objects, therefore disable assignment operator
        ConvolutionLayer& operator=(const ConvolutionLayer& ) = delete;   

        // constructor 
        ConvolutionLayer(const int kernelH, const int kernelW, const int strideH, const int strideW, const int padH, const int padW, const bool padModeSame);
        
        // This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
        virtual int validateInput() const;

        // This method should validate output parameters, and return TRUE if everything is ok, FALSE otherwise        
        virtual int validateOutput() const;

        // This method should validate parameters & bias, and return TRUE if everything ok. False otherwise
        virtual int validateParameters() const;

        // this method should validate memory/holders for BP pass
        virtual int validateGradients() const;

        // feed forward
        virtual int feedForward();

        // back propagate
        virtual int backPropagate();
        
        // destructor
        ~ConvolutionLayer();
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

    _extraParams = new T[7];

    _extraParams[0] = _kernelW;
    _extraParams[1] = _kernelH;
    _extraParams[2] = _strideW;
    _extraParams[3] = _strideH;
    _extraParams[4] = _padW;
    _extraParams[5] = _padH;
    _extraParams[6] = _padModeSame ? 1.0 : 0.0;
}

//////////////////////////////////////////////////////////////////////
// This method should validate input parameters, and return TRUE if everything ok. FALSE otherwise
template<typename T, typename AF> int ConvolutionLayer<T,AF>::validateInput() const {
    
   if (this->_input == nullptr || !this->_input->nonNull())        
        return ND4J_STATUS_BAD_INPUT;

    if (this->_input->rankOf() != 4)
        return ND4J_STATUS_BAD_RANK;

    if (this->_input->getShapeInfo()[2] != this->_params->getShapeInfo()[2])
        return ND4J_STATUS_BAD_SHAPE;        
    
}

//////////////////////////////////////////////////////////////////////
// This method should valudate output parameters, and return TRUE if everything is ok, FALSE otherwise
template<typename T, typename AF> int ConvolutionLayer<T,AF>::validateOutput() const {
    
    if (this->_output == nullptr || !this->_output->nonNull())
        return ND4J_STATUS_BAD_OUTPUT;

    if (this->_output->rankOf() != 4)
        return ND4J_STATUS_BAD_RANK;

    if (this->_output->getShapeInfo()[1] != this->_input->getShapeInfo()[1] || this->_output->getShapeInfo()[2] != this->_params->getShapeInfo()[1])
        return ND4J_STATUS_BAD_OUTPUT;
    
    float oH = (this->_input->getShapeInfo()[3] - _kernelH + 2.*_padH) / (_strideH + 1.);
    // oH must be integer ! 
    if (oH != (int)oH)
        return ND4J_STATUS_BAD_SHAPE;
    float oW = (this->_input->getShapeInfo()[4] - _kernelW + 2.*_padW) / (_strideW + 1.);
    // oW must be integer ! 
    if (oW != (int)oW)
        return ND4J_STATUS_BAD_SHAPE;
        
    return ND4J_STATUS_OK;
}

//////////////////////////////////////////////////////////////////////
template<typename T, typename AF> int ConvolutionLayer<T,AF>::validateParameters() const {
    
    if (this->_params == nullptr || !this->_params->nonNull() || this->_bias == nullptr || !this->_bias->nonNull())
        return ND4J_STATUS_BAD_PARAMS;    
    
    if (this->_params->rankOf() != 4 || this->_bias->rankOf() != 2)
        return ND4J_STATUS_BAD_RANK;

    // check the weight matrix consistency with class kernel members 
    if (this->_params->getShapeInfo()[3] != _kernelH || this->_params->getShapeInfo()[4] != _kernelW)
        return ND4J_STATUS_BAD_PARAMS;

    if (this->_params->getShapeInfo()[1] != this->_bias->getShapeInfo()[2])
        return ND4J_STATUS_BAD_SHAPE;
    
    return ND4J_STATUS_OK;
}


//////////////////////////////////////////////////////////////////////
template<typename T, typename AF> int ConvolutionLayer<T,AF>::validateGradients() const {

    // if (this->_gradientW == nullptr || this->_gradientB == nullptr ||  !this->_gradientW->nonNull() || !this->_gradientB->nonNull())
        // return ND4J_STATUS_BAD_GRADIENTS;    

    // if (this->_epsilonNext == nullptr || !this->_epsilonNext->nonNull())
        // return ND4J_STATUS_BAD_OUTPUT;
        
    // if (!this->_gradientW->isSameShape(*this->_params)) 
        // return ND4J_STATUS_BAD_GRADIENTS;
    
    // if (!this->_gradientB->isSameShape(*this->_bias))
        // return ND4J_STATUS_BAD_BIAS;

    // // we're checking equality of input/epsilon batch size
    // if (this->_epsilon->shapeOf()[0] != this->_input->shapeOf()[0])
        // return ND4J_STATUS_BAD_EPSILON;

    // if (this->_epsilon->columns() != this->_bias->columns())
        // return ND4J_STATUS_BAD_EPSILON;

    // // batch comparison again
    // if (!this->_epsilonNext->isSameShape(*this->_input))
        // return ND4J_STATUS_BAD_OUTPUT;

    // return ND4J_STATUS_OK;
}

//////////////////////////////////////////////////////////////////////
// feed forward
template<typename T, typename AF> int ConvolutionLayer<T,AF>::feedForward() {
   
    const int bS = this->_input->getShapeInfo()[1];     // batch size, number of examples
    const int iD = this->_input->getShapeInfo()[2];     // input depth
    const int oD = this->_output->getShapeInfo()[2];    // output depth
    const int oH = this->_output->getShapeInfo()[3];    // output height
    const int oW = this->_output->getShapeInfo()[4];    // output width
    // create temporary 6D array for the needs of Im2col, it will serve as output array there
    NDArray<T> arr6d('f', {bS, iD, _kernelH, _kernelW, oH, oW});
    // call Im2col
    functions::transform::Transform<T>::template 
    exec<simdOps::Im2col<T>>(this->_input->getBuff(), this->_input->getShapeInfo(), arr6d.getBuff(), 
                             arr6d.getShapeInfo(), _extraParams, nullptr, nullptr);
    
    if(!arr6d.reshape({bS*oH*oW, iD*_kernelH*_kernelW}))
        return ND4J_STATUS_BAD_SHAPE;
    // prepare _output, reshape to 2D     
    this->_output->replacePointers(nullptr,nullptr);    
    this->_output->setShape('f',{bS*oH*oW, oD});    
    // reshape _params to 2D    
    if(this->_params->reshape({iD*_kernelW*_kernelH, oD}))
        return ND4J_STATUS_BAD_PARAMS;
    // Z = IW
    this->gemmHelper(&arr6d, this->_params, this->_output, (T) 1.0f, (T) 0.0f);
    // Z += B
    this->_output->addiRowVector(this->_bias);
    // reshape _params and output  back to 4D
    this->_output->reshape({bS, oD, oH, oW});
    this->_params->reshape({oD, iD, _kernelH, _kernelW});
    // apply activations F(Z)
    ActivationsExecutioner<T>::template executeFF<AF>(this->_output, this->_output);
    
    return ND4J_STATUS_OK;
}

// back propagation
template<typename T, typename AF> int ConvolutionLayer<T,AF>::backPropagate() {

} 


////////////////////////////////////////////////////////////////////////
// destructor
template<typename T, typename AF> ConvolutionLayer<T,AF>::~ConvolutionLayer() {
    
    delete []_extraParams;
}



// end of namespace brackets
}
}


#endif //PROJECT_CONVOLUTIONLAYER_H