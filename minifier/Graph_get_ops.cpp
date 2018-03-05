//
// @author raver119@gmail.com
// added methon getOperations by GS <sgazeos@gmail.com>
//

#include <graph/Graph.h>
#include <helpers/EnumUtils.h>
#include <graph/FlatUtils.h>
#include <NativeOps.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        std::vector<OpDescriptor> Graph<T>::getOperations() const {
            buildGraph();
            nd4j_printf("\nRetrieving ops from the Graph...\n", "");
            std::vector<OpDescriptor> res;

            int opCnt = 0;
            for (int l = 0; l < _onion->size(); l++) {
                int layerSize = _onion->count(l) == 1 ? _onion->at(l)->size() : 0;

                for (int n = 0; n < layerSize; n++) {
                    Node<T>* node = _onion->at(l)->at(n);
                    if (node->name() == nullptr) continue;
                    std::string* opName = node->name();
                    int numInputs = 0;
                    int numOutputs = 0;

                    if (node->inputs())
                        numInputs = node->inputs()->size();

                    if (node->outputs())
                        numOutputs = node->outputs()->size();
                    bool inplace = node->isInplace();

                    OpDescriptor opDescriptor(numInputs, numOutputs, *opName, inplace);

                    // we're skipping Scopes here
                    if (node->opType() == OpType_LOGIC && node->opNum() == 10)
                        continue;

                    //printOutNode(node);
                    res.emplace_back(opDescriptor);
                }
            }


            nd4j_printf("\nPrinting out Scopes...\n","");
            for (int s = 0; s < _scopes.size(); s++) {
                Scope<T>* scope = _scopes.at(s);
                nd4j_printf("Scope %i:<%s>:\n", scope->id(), scope->name()->c_str());

                for (int n = 0; n < scope->nodes()->size(); n++) {
                    Node<T>* node = scope->nodes()->at(n);
                    //printOutNode(node);
                    if (node->name() == nullptr) continue;
                    std::string* opName = node->name();
                    int numInputs = 0;
                    int numOutputs = 0;

                    if (node->inputs())
                        numInputs = node->inputs()->size();

                    if (node->outputs())
                        numOutputs = node->outputs()->size();
                    bool inplace = node->isInplace();

                    OpDescriptor opDescriptor(numInputs, numOutputs, *opName, inplace);

                    res.emplace_back(opDescriptor);

                }
            }

            return res;
        }
    }
}

