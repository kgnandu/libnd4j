//
//  @author raver119@gmail.com
//

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/ops.h>
#include <vector>
#include <NDArray.h>
#include <NDArrayFactory.h>

template<typename T>
nd4j::NDArray<T>  * processCondition(int mode,nd4j::NDArray<T> *arg, nd4j::NDArray<T> *comp, T compScalar);

template <typename T>
T processElementCondition(int mode,T d1,T d2);




template<typename T>
nd4j::NDArray<T>  * processCondition(int mode,nd4j::NDArray<T> *arg, nd4j::NDArray<T> *comp, T compScalar) {
    std::vector<T> result;
    if(comp != nullptr) {
        if (comp->isScalar()) {
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            nd4j::NDArray<T> arg1 = *arg;
            nd4j::NDArray<T> comp1 = *comp;
            for (Nd4jIndex i = 0; i < arg->lengthOf(); i++) {
                T result2 = processElementCondition<T>(mode,arg1(i),comp1(i));
                if(result2 > 0)
                    result.push_back(arg1(i));
            }
        } else {
            // REQUIRE_TRUE(comp.isSameShape(arg));
            //Other input for compare could be an ndarray or a secondary scalar
            //for comparison
            nd4j::NDArray<T> arg1 = *arg;
            for (Nd4jIndex i = 0; i < arg->lengthOf(); i++) {
                T result2 = processElementCondition<T>(mode,arg1(i),compScalar);
                if(result2 > 0)
                    result.push_back(arg1(i));
            }
        }

    }
    else {
        nd4j::NDArray<T> arg1 = *arg;
        //Other input for compare could be an ndarray or a secondary scalar
        //for comparison
        for (Nd4jIndex i = 0; i < arg->lengthOf(); i++) {
            T result2 = processElementCondition<T>(mode,arg1(i),compScalar);
            if(result2 > 0)
                result.push_back(arg1(i));
        }
    }

    std::vector<int> shape;
    shape.push_back(result.size());
    nd4j::NDArray<T> *ret = new nd4j::NDArray<T>(result.data(),'c',shape,arg->getWorkspace());
    nd4j_printf("Choose returning size %d array\n",result.size());
    return ret;

}

template <typename T>
T processElementCondition(int mode,T d1,T d2) {
    T modePointer = (T ) mode;
    T input[3] = {d2,EPS,mode};
    T res = simdOps::MatchCondition<T>::op(d1, input);
    return res;

}


namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(choose, -1, 1, false, -1, -1) {
            int mode = INT_ARG(0);
            if (block.width() > 1) {
                auto arg = INPUT_VARIABLE(0);
                auto comp = INPUT_VARIABLE(1);
                auto result = processCondition<T>(mode,arg,comp,0.0f);
                OVERWRITE_RESULT(result);
            }//scalar case
            else {
                nd4j_printf("In scalar case %d\n",1);
                T scalar = (T) T_ARG(0);
                auto arg = INPUT_VARIABLE(0);
                auto  result = processCondition<T>(mode,arg,nullptr,scalar);
                OVERWRITE_RESULT(result);
            }


            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(choose) {
            
            return new ShapeList();
        }


    }
}