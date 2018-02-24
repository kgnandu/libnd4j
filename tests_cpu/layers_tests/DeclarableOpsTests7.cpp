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
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    NDArray<double> scalar('c',{1,1},{0.0});
    nd4j::ops::choose<double> op;
   //greater than test
    auto result = op.execute({&x,&scalar}, {1.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(4,z->lengthOf());
    delete z;
    //ASSERT_TRUE(exp.isSameShape(z));

    delete[] result;
}