//
// Created by raver119 on 27/10/17.
//

#include <op_boilerplate.h>
#include <memory>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpRegistrator.h>
#include <declarable/generic/helpers/convolutions.h>


namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(deconv2d, 2, 1, false, 0, 9) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;
            if (block.getVariables()->size() > 2)
                bias = INPUT_VARIABLE(2);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead", weights->rankOf());

            int oD = weights->sizeAt(0);

            if (bias != nullptr) {
                REQUIRE_TRUE(bias->isVector(), 0, "Bias should be vector");
                REQUIRE_TRUE(bias->lengthOf() == oD, 0, "Bias length be equal to outpuDepth, but got %i instead", bias->lengthOf());
            }

            int iY = input->sizeAt(2);
            int iX = input->sizeAt(3);

            int kY = block.getIArguments()->at(0);
            int kX = block.getIArguments()->at(1);
            int sY = block.getIArguments()->at(2);
            int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            int dY = block.getIArguments()->at(6);
            int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            NDArray<T> *z = this->getZ(block);

            int oY = z->sizeAt(2);
            int oX = z->sizeAt(3);

            auto gcol = nd4j::NDArrayFactory<T>::tensorDot(weights, input, nullptr, {0}, {1});
            gcol->permutei({3, 0, 1, 2, 4, 5});

            std::vector<T> extrasCol2Im({(T) sY, (T) sX, (T) pY, (T) pX, (T) oY, (T) oX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            gcol->template applyTransform<simdOps::Col2Im<T>>(z, extrasCol2Im.data());

            delete gcol;

            if (bias != nullptr) {
                z->template applyBroadcast<simdOps::Add<T>>({1}, bias);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(deconv2d) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            int B = shape::shapeOf(inShape)[0];
            int iC = shape::shapeOf(inShape)[1];
            int iY = shape::shapeOf(inShape)[2];
            int iX = shape::shapeOf(inShape)[3];

            int oC = shape::shapeOf(wShape)[0];
            int kY = block.getIArguments()->at(0);
            int kX = block.getIArguments()->at(1);
            int sY = block.getIArguments()->at(2);
            int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            int dY = INT_ARG(6);
            int dX = INT_ARG(7);

            int oY = sY * (iY - 1) + kY - 2 * pY;
            int oX = sX * (iX - 1) + kX - 2 * pX;

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
            std::vector<int> shape({B, oC, oY, oX});
            shape::shapeBuffer(4, shape.data(), newShape);

            return new ShapeList(newShape);
        }


        CUSTOM_OP_IMPL(deconv2d_bp, 4, 2, false, 0, 9) {
            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *weights = INPUT_VARIABLE(1);
            NDArray<T> *epsilonNext = INPUT_VARIABLE(2);
            NDArray<T> *bias = nullptr;

            // bias is still optional
            if (block.getVariables()->size() > 3)
                bias = INPUT_VARIABLE(3);

            //epsilonNext->rankOf() == 4 && weights->rankOf() == 4
            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead", weights->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "Epsilon should be 4D, but got %iD instead", epsilonNext->rankOf());

            int kY = block.getIArguments()->at(0);
            int kX = block.getIArguments()->at(1);
            int sY = block.getIArguments()->at(2);
            int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            int dY = block.getIArguments()->at(6);
            int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            NDArray<T>* epsilon = this->getZ(block);
            NDArray<T>* gradW = this->getZ(block, 1);
            NDArray<T>* gradB = nullptr;

            if (bias != nullptr)
                gradB = this->getZ(block, 2);

            // epsilon for deconv2d is FF conv pass

            nd4j::ops::conv2d<T> op;
            Nd4jStatus r1 = op.execute({input, weights}, {epsilon}, {}, {kY, kX, sY, sX, pY, pX, dY, dX, block.getIArguments()->at(8)});
            if (r1 != ND4J_STATUS_OK)
                return r1;

            // gradW is im2col + tensorDot
            /*
              col = conv.im2col_cpu(
            x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)
             gW = numpy.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))).
             */

            int oY = 0;
            int oX = 0;
            int inY = epsilonNext->sizeAt(2);
            int inX = epsilonNext->sizeAt(3);

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);
            }

            std::vector<T> extrasIm2Col({(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});
            auto gcol = new NDArray<T>('c', {input->sizeAt(0), input->sizeAt(1), kY, kX, oY, oX });
            epsilonNext->template applyTransform<simdOps::Im2col<T>>(gcol, extrasIm2Col.data());

            /*
            gW = numpy.tensordot(
                    gy, col, ((0, 2, 3), (0, 4, 5))).
            */

            auto gW = NDArrayFactory<T>::tensorDot(input, gcol, nullptr, {0, 2, 3}, {0, 4, 5});
            gradW->assign(gW);

            delete gW;
            delete gcol;

            if (gradB != nullptr) {
                auto sum = epsilon->template reduceAlongDimension<simdOps::Sum<T>>({0, 2, 3});
                gradB->assign(sum);
                delete sum;

                STORE_3_RESULTS(*epsilon, *gradW, *gradB);
            } else {
                STORE_2_RESULTS(*epsilon, *gradW);
            }



            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(deconv2d_bp) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);
            auto eShape = inputShape->at(2);
            int *bShape = nullptr;

            // bias is optional thing, and might be absent
            if (inputShape->size() == 4)
                bShape = inputShape->at(3);

            int *newInShape;
            int *newWShape;
            ALLOCATE(newInShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            ALLOCATE(newWShape, block.getWorkspace(), shape::shapeInfoLength(wShape), int);

            memcpy(newInShape, inShape, shape::shapeInfoByteLength(inShape));
            memcpy(newWShape, wShape, shape::shapeInfoByteLength(wShape));

            auto shapes = new ShapeList({newInShape, newWShape});

            if (bShape != nullptr) {
                int *newBShape;
                ALLOCATE(newBShape, block.getWorkspace(), shape::shapeInfoLength(bShape), int);
                memcpy(newBShape, bShape, shape::shapeInfoByteLength(bShape));

                shapes->push_back(newBShape);
            }

            return shapes;
        }
    }
}
