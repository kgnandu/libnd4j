//
// Created by GS <sgazeos@gmail.com> on 05.04.18.
//

#ifndef __DYNAMIC_H_HELPERS__
#define __DYNAMIC_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            void dynamicPartitionFunctor(NDArray<T>* input, NDArray<T>* indices, std::vector<NDArray<T>*>& outputList);
        }
    }
}
#endif
