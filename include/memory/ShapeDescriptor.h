//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_SHAPE_DESCRIPTOR_H
#define LIBND4J_SHAPE_DESCRIPTOR_H

#include <initializer_list>
#include <vector>
#include <pointercast.h>

namespace nd4j {
namespace memory {
    class ShapeDescriptor {
    private:
        char _rank;
        char _order;
        std::vector<int> _shape;
        std::vector<int> _strides;
        int _elementWiseStride = 1;
        int _offset = 0;

        int *_bufferH = nullptr;
        int *_bufferD = nullptr;

    public:
        explicit ShapeDescriptor(int *shapeInfoH, int *shapeInfoD = nullptr);
        ShapeDescriptor(std::initializer_list<int> shape, char order = 'c');
        ShapeDescriptor(std::vector<int> &shape, char order = 'c');

        ~ShapeDescriptor() = default;

        Nd4jIndex hash();
    };
}
}

#endif