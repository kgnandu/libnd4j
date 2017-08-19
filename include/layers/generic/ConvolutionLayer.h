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

    public:
        virtual int feedForward() {}
        virtual int backPropagate() {} 
};



//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///////////////////// implementation part ////////////////////////////

// feed forward
// template<typename T, typename AF> int ConvolutionLayer<T,AF>::feedForward() {
   
    // // gemm here, input * W
    // // these values should be set appropriately

    // this->gemmHelper(this->_input, this->_params, this->_output, (T) 1.0f, (T) 0.0f);

    // // we're rolling through rows here
    // this->_output->addiRowVector(this->_bias);
   

    // // activation call
    // ActivationsExecutioner<T>::template executeFF<AF>(this->_output, this->_output);

    // return ND4J_STATUS_OK;
// }

// end of namespace brackets
}
}

#endif //PROJECT_CONVOLUTIONLAYER_H
