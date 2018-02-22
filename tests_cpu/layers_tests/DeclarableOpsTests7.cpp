//
// Created by raver119 on 09.02.18.
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests7 : public testing::Test {
public:

    DeclarableOpsTests7() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR) {
    NDArray<double> x('c',{1,4},double[4]{1,2,3,4});
    nd4j::ops::choose<double> op;

    auto result = op.execute({&x}, {}, {1, 1,5,7,1, 1,2,3,1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}