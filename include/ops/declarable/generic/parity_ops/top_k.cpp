//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(top_k, 1, 2, false, -2, 0) {
            NDArray<T>* x = INPUT_VARIABLE(0);
            int k = 1;// from params
            bool needSort = true;
//            NDArray<T>* values = OUTPUT_VARIABLE(0);
//            NDArray<int>* indeces = OUTPUT_VARIABLE(1);
            if (block.getIArguments()->size() > 0) {
                k = INT_ARG(0);
                needSort = INT_ARG(1);
            }

            if (k == 1) {
                // using arg_max for it
                NDArray<T>* values = OUTPUT_VARIABLE(0);
                auto indeces = OUTPUT_VARIABLE(1);
                //int index = 0;
                //x->template applyTransform<nd4j::ops::argmax<T>>(indeces);
                x->template applyIndexReduce<simdOps::IndexMax<T>>(indeces, {});

                int index = indeces->getScalar(0);
                T val = x->getScalar(index);
                //OVERWRITE_2_RESULTS(vals, indeces);
                
                values->putScalar(val, 0);
                //indeces->putScalar(index, 0);
                return ND4J_STATUS_OK;
            }
            else if (k > 1) {
                std::vector<int> inds(k);
                std::vector<T> vals(k);
                //indeces = x->template indexReduceNumber<simdOps::IndexMax<T>>(x->shape()->dimension());
                for (int e = 0; e < k; e++) {
                    vals[e] = x->getScalar(e); // start initializing
                    inds[e] = e;
                }

                for (int e = k; e < x->lengthOf(); e++) {
                    T v = x->getScalar(e);
                    for (int j = 0; j < k; ++j) {
                        if (v > vals[j]) {
                            vals[j] = v;
                            inds[j] = e;
                            break; // enough
                        }
                    }
                }
            
                // if need to be sort results
            if (needSort) {
                std::sort(inds.begin(), inds.end(), [vals](int a, int b) {
                    return vals[a] > vals[b];   
                });
            }
            auto vec_vals = new NDArray<T>('c', {1, k});
            auto vec_idx = new NDArray<T>('c', {1, k});

            for (int e = 0; e < k; e++) {
                vec_vals->putScalar(e, vals.at(e));
                vec_idx->putScalar(e, inds.at(e));
            }

                OVERWRITE_2_RESULTS(vec_vals, vec_idx);
                return ND4J_STATUS_OK;
            }
            else
                return ND4J_STATUS_BAD_ARGUMENTS;
        }

        DECLARE_SHAPE_FN(top_k) {
            auto shapeList = new ShapeList(); 

            int k = 1; // default output shape is size 1

            if (block.getIArguments()->size() > 0) {
                k = INT_ARG(0);
            }

            for (int e = 0; e < 2; e++) { // 2 element tuple at output
                int* newshape;
                ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(k), int);
                //std::vector<int> shape(k);
                //{1, 1, k, 1, 1, 0, 1, 99};
                shape::shapeBuffer(k, newshape);
                shapeList->push_back(newshape); 
            }
            return shapeList;
        }

    }
}