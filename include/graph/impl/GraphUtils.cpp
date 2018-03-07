//
// Created by GS <sgazeos@gmail.com> 3/7/2018
//

#include <graph/GraphUtils.h>

namespace nd4j {
namespace graph {

bool 
GraphUtils::filterOperations(GraphUtils::OpList& ops) {
    bool modified = false;

    std::vector<OpDescriptor>& filtered(ops);

    std::sort(filtered.begin(), filtered.end(), [](OpDescriptor a, OpDescriptor b) {
        return a.getOpName()->compare(*(b.getOpName())) > 0;
    });
    std::string name = *(filtered[0].getOpName());

    for (int x = 1; x < filtered.size(); x++) {
        if (filtered[x].getOpName()->compare(name) == 0) {
            // there is a match
            auto fi = std::find_if(ops.begin(), ops.end(), 
                [name](OpDescriptor a) { 
                    return a.getOpName()->compare(name) == 0; 
            });
            ops.erase(fi);
            modified = true;
        }
        name = *(filtered[x].getOpName());
    }
    return modified;
}

std::string 
GraphUtils::makeCommandLine(GraphUtils::OpList& ops) {
    std::string res;

    if (!ops.empty()) {
        res += std::string("\n ./buildnativeoperations.sh -g \"-D_"); 
        res += *(ops[0].getOpName());
        for (int i = 1; i < ops.size(); i++) {
            res += std::string(";-D_");
            res += *(ops[i].getOpName());
        }
        res += '\"';
    }

    return res;
}

}
}
