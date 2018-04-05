//
// Created by george on 05.04.18.
//
#include <ops/declarable/helpers/dynamic.h>
#include <NDArrayFactory.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            void dynamicPartitionFunctor(NDArray<T>* input, NDArray<T>* indices, std::vector<NDArray<T>*>& outputList) {
                std::vector<std::pair<NDArray<T> *, int>> outputs(outputList.size());
                if (input->rankOf() != indices->rankOf()) {
                    std::vector<int> sourceDims(input->rankOf() - indices->rankOf());

                    for (int i = sourceDims.size(); i > 0; i--)
                        sourceDims[sourceDims.size() - i] = input->rankOf() - i;

                    std::unique_ptr<ResultSet<T>> listOfTensors(
                            NDArrayFactory<T>::allTensorsAlongDimension(input, sourceDims));
                    for (unsigned int i = 0; i < outputList.size(); i++) {
                        outputs[i].first = outputList[i];
                        std::vector<int> outDims(outputs[i].first->rankOf() - 1);
                        for (int k = 1; k < outputs[i].first->rankOf(); k++)
                            outDims[k - 1] = k;
                        std::unique_ptr<ResultSet<T>> listOutForCurrent(
                                NDArrayFactory<T>::allTensorsAlongDimension(outputs[i].first, outDims));
                        outputs[i].second = 0;
                        for (int e = 0; e < indices->lengthOf(); ++e)
                            if ((*indices)(e) == T(i))
                                listOutForCurrent->at(outputs[i].second++)->assign(listOfTensors->at(e));
                    }

                } else
                    for (unsigned int i = 0; i < outputList.size(); i++) {
                        outputs[i].first = outputList[i];
                        outputs[i].second = 0;
                        for (int e = 0; e < indices->lengthOf(); ++e)
                            if ((*indices)(e) == T(i))
                                outputs[i].first->putScalar(outputs[i].second++, (*input)(e));
                    }
            }

            template void dynamicPartitionFunctor(NDArray<float>* input, NDArray<float>* indices, std::vector<NDArray<float>*>& outputList);
            template void dynamicPartitionFunctor(NDArray<float16>* input, NDArray<float16>* indices, std::vector<NDArray<float16>*>& outputList);
            template void dynamicPartitionFunctor(NDArray<double>* input, NDArray<double>* indices, std::vector<NDArray<double>*>& outputList);
        }
    }
}

