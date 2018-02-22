//
//  @author raver119@gmail.com
//

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/ops.h>
#include <vector>
#include <NDArray.h>

template<typename T>
nd4j::NDArray<T>  processCondition(int mode,nd4j::NDArray<T> *arg, nd4j::NDArray<T> *comp, T compScalar);

template <typename T>
T processElementCondition(int mode,T d1,T d2);




template<typename T>
nd4j::NDArray<T>  processCondition(int mode,nd4j::NDArray<T> *arg, nd4j::NDArray<T> *comp, T compScalar) {
    std::vector<T> result;
    if(comp != nullptr) {
        if (comp->isScalar()) {
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            for (Nd4jIndex i = 0; i < arg->lengthOf(); i++) {
                result.push_back(processElementCondition<T>(mode,arg[i],comp[i]));
            }
        } else {
           // REQUIRE_TRUE(comp.isSameShape(arg));
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            for (Nd4jIndex i = 0; i < arg->lengthOf(); i++) {
                result.push_back(processElementCondition<T>(mode,arg[i],compScalar));
            }
        }

    }
    else {
        //Other input for compare could be an ndarray or a secondary scalar
        //for comparison
        for (Nd4jIndex i = 0; i < arg->lengthOf(); i++) {
            result.push_back(processElementCondition<T>(mode,arg[i],compScalar));
        }
    }

    std::vector<int> shape;
    shape.push_back(result.size());
    nd4j::NDArray<T> ret(result.data(),'c',shape,arg->getWorkspace());
    return ret;

}

template <typename T>
T processElementCondition(int mode,T d1,T d2) {
    T modePointer = (T ) mode;
    T input[2] = {mode,d2};
    T res = simdOps::MatchCondition<T>::op(d1, input);
    return res;

}


namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(choose, -1, 1, false, 0, 1) {
            int mode = INT_ARG(0);
            if (block.width() > 1) {
                auto arg = INPUT_VARIABLE(0);
                auto comp = INPUT_VARIABLE(1);
                auto result = processCondition<T>(mode,arg,comp,0.0f);
                OVERWRITE_RESULT(result);
            }//scalar case
            else {
                T scalar = (T) T_ARG(0);
                auto arg = INPUT_VARIABLE(0);
                auto  result = processCondition<T>(mode,arg,nullptr,scalar);
                OVERWRITE_RESULT(result);
            }


            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(choose) {
            auto inShape = inputShape->at(1);

            int *newshape;
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newshape, inShape, shape::shapeInfoByteLength(inShape));

            return new ShapeList(newshape);
        }


    }
}