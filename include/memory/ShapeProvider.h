//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_SHAPE_PROVIDER_H
#define LIBND4J_SHAPE_PROVIDER_H

#include <helpers/shape.h>
#include <memory/ShapeDescriptor.h>
#include <map>
#include <pointercast.h>

namespace nd4j {
namespace memory {
    class ShapeProvider {
    private:
        std::map<Nd4jIndex, ShapeDescriptor> _hostCache;

        ShapeProvider() = default;
        ~ShapeProvider() = default;
    public:



        int *reshape();
        int *permute();
    };
}
}

#endif