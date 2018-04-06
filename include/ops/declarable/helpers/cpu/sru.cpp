//
// implementation of operations for Simple Recurrent Unit: arXiv:1709.02755v2 [cs.CL] 12 Sep 2017
//
//  @author Yurii Shyrma, created on 05.12.2017
//

#include<ops/declarable/helpers/sru.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> activation(const NDArray<T>& arr) {    
    
    // return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
    NDArray<T> result(&arr, false, arr.getWorkspace());
    (const_cast<NDArray<T>&>(arr)).template applyTransform<simdOps::Tanh<T>>(&result);    
    return result;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> sigmoid(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Sigmoid<T>>();    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void sruCell(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {

    NDArray<T>* xt   = inArrs[0];               // input [bS x inSize], bS - batch size, inSize - number of features
    NDArray<T>* ct_1 = inArrs[1];               // previous cell state ct  [bS x inSize], that is at previous time step t-1   
    NDArray<T>* w    = inArrs[2];               // weights [inSize x 3*inSize]
    NDArray<T>* b    = inArrs[3];               // biases [2*inSize]

    NDArray<T>* ht   = outArrs[0];              // current cell output [bS x inSize], that is at current time step t
    NDArray<T>* ct   = outArrs[1];              // current cell state  [bS x inSize], that is at current time step t

    const int bS     = xt->sizeAt(0);    
    const int inSize = xt->sizeAt(1);           // inSize - number of features
            
    NDArray<T> z = mmul(*xt, *w);               //  [bS x 3*inSize]    

    // forget gate = sigmoid(xt*Wf + bf)
    NDArray<T> ft = sigmoid<T>(z({{},{inSize,   2*inSize}}) + (*b)({{0, inSize}}));
    
    // reset gate = sigmoid(xt*Wr + br)
    NDArray<T> rt = sigmoid<T>(z({{},{2*inSize, 3*inSize}}) + (*b)({{inSize, 2*inSize}}));

    // current sell state = ft(*)ct_1 + (1 - ft)(*)(*)(xt*Wc)
    ct->assign( ft*(*ct_1) + ((T)1. - ft) * z({{},{0, inSize}}) );
    // *ct = ft*(*ct_1 - z({},{0, inSize})) + z({{},{0, inSize}});

    // current cell output = rt(*)activation(ct) + (1 - rt)(*)xt
    ht->assign( rt*activation<T>(*ct) + ((T)1. - rt) * (*xt) );    
    // *ht = rt * (activation<T>(ct) - *xt) + *xt;        
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void sruTimeLoop(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {
    
    NDArray<T>* x  = inArrs[0];                     // input [bS x inSize x time]
    NDArray<T>* c0 = inArrs[1];                     // initial cell state  (at time step = 0) [bS x inSize],  
    NDArray<T>* w  = inArrs[2];                     // weights, [3*inSize x inSize] 
    NDArray<T>* b  = inArrs[3];                     // biases,  [2*inSize] 
    
    NDArray<T>* h  = outArrs[0];                    // cell outputs [bS x inSize x time]
    NDArray<T>* c  = outArrs[1];                    // cell states  [bS x inSize x time]

    w = w->transpose();                             // [3*inSize x inSize] -> [inSize x 3*inSize] 

    const int time  = x->sizeAt(2);

    NDArray<T> ct_1(c0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        NDArray<T> xt = (*x)({{}, {}, {t,t+1}});
        NDArray<T> ht = (*h)({{}, {}, {t,t+1}});
        NDArray<T> ct = (*c)({{}, {}, {t,t+1}});

        helpers::sruCell<T>({&xt, &ct_1, w, b},  {&ht, &ct});        
        ct_1.assign(ct);
    }    

    delete w;
}




template void sruCell<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs);
template void sruCell<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs);
template void sruCell<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs);

template void sruTimeLoop<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs);
template void sruTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs);
template void sruTimeLoop<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs);


}
}
}