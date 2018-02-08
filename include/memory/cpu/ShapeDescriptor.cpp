//
//  @author raver119@gmail.com
//

#include <memory/ShapeDescriptor.h>
#include <helpers/shape.h>

namespace nd4j {
namespace memory {
    ShapeDescriptor::ShapeDescriptor(int *shapeInfo, int *shapeInfoD) {
        _rank = shape::rank(shapeInfo);
        _order = shape::order(shapeInfo);
        _elementWiseStride = shape::elementWiseStride(shapeInfo);
        _offset = shape::offset(shapeInfo);

        _shape.resize(_rank);
        _strides.resize(_rank);

        for (int e = 0; e < _rank; e++)
            _shape[e] = shape::shapeOf(shapeInfo)[e];

        for (int e = 0; e < _rank; e++)
            _strides[e] = shape::stride(shapeInfo)[e];

        _bufferH = shapeInfo;
        _bufferD = shapeInfoD;
    };

    ShapeDescriptor::ShapeDescriptor(std::initializer_list<int> shape, char order) {
        std::vector<int> tmp(shape);

        _shape = tmp;
        _order = order;
    };

    ShapeDescriptor::ShapeDescriptor(std::vector<int> &shape, char order) {
        _shape = shape;
        _order = order;
    };

    Nd4jIndex ShapeDescriptor::hash() {
        return 0L;
    }
}
}