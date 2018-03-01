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


TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_LARGE) {
    double inputData[150] = {
            0.00,  0.51,  0.68,  0.69,  0.86,  0.91,  0.96,  0.97,  0.97,  1.03,  1.13,  1.16,  1.16,  1.17,  1.19,  1.25,  1.25,  1.26,  1.27,  1.28,  1.29,  1.29,  1.29,  1.30,  1.31,  1.32,  1.33,  1.33,  1.35,  1.35,  1.36,  1.37,  1.38,  1.40,  1.41,  1.42,  1.43,  1.44,  1.44,  1.45,  1.45,  1.47,  1.47,  1.51,  1.51,  1.51,  1.52,  1.53,  1.56,  1.57,  1.58,  1.59,  1.61,  1.62,  1.63,  1.63,  1.64,  1.64,  1.66,  1.66,  1.67,  1.67,  1.70,  1.70,  1.70,  1.72,  1.72,  1.72,  1.72,  1.73,  1.74,  1.74,  1.76,  1.76,  1.77,  1.77,  1.80,  1.80,  1.81,  1.82,  1.83,  1.83,  1.84,  1.84,  1.84,  1.85,  1.85,  1.85,  1.86,  1.86,  1.87,  1.88,  1.89,  1.89,  1.89,  1.89,  1.89,  1.91,  1.91,  1.91,  1.92,  1.94,  1.95,  1.97,  1.98,  1.98,  1.98,  1.98,  1.98,  1.99,  2.00,  2.00,  2.01,  2.01,  2.02,  2.03,  2.03,  2.03,  2.04,  2.04,  2.05,  2.06,  2.07,  2.08,  2.08,  2.08,  2.08,  2.09,  2.09,  2.10,  2.10,  2.11,  2.11,  2.11,  2.12,  2.12,  2.13,  2.13,  2.14,  2.14,  2.14,  2.14,  2.15,  2.15,  2.16,  2.16,  2.16,  2.16,  2.16,  2.17
    };

    NDArray<double> x(inputData,'c',{1,149});
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x}, {0.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(1);
    auto array = *z;
    ASSERT_EQ(148,array(0));
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}

TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_ZERO) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x}, {0.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(1);
    auto array = *z;
    ASSERT_EQ(3,array(0));
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}


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
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}


TEST_F(DeclarableOpsTests7, Test_CHOOSE_SCALAR_LEFT) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    NDArray<double> scalar('c',{1,1},{0.0});
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&scalar,&x}, {1.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}


TEST_F(DeclarableOpsTests7, Test_CHOOSE_ONLY_SCALAR) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x}, {1.0},{3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}


TEST_F(DeclarableOpsTests7, Test_CHOOSE_ONLY_SCALAR_GTE) {
    std::vector<double> data;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
    }



    NDArray<double> x('c',{1,4},data);
    nd4j::ops::choose<double> op;
    //greater than test
    auto result = op.execute({&x}, {1.0},{5});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));

    delete result;

}





TEST_F(DeclarableOpsTests7, TEST_WHERE) {
    std::vector<double> data;
    std::vector<double> mask;
    std::vector<double> put;
    std::vector<double> resultData;
    std::vector<double> assertion;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
        if(i >  1) {
            assertion.push_back(5.0);
            mask.push_back(1);
        }
        else {
            assertion.push_back(i);
            mask.push_back(0);
        }

        put.push_back(5.0);
        resultData.push_back(0.0);
    }




    NDArray<double> x('c',{1,4},data);
    NDArray<double> maskArr('c',{1,4},mask);
    NDArray<double> putArr('c',{1,4},put);
    NDArray<double> resultArr('c',{1,4},resultData);
    nd4j::ops::where<double> op;
    //greater than test
    //            Nd4jStatus execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);

    auto result = op.execute({&maskArr,&x,&putArr},{&resultArr}, {},{3},false);
    // ASSERT_EQ(Status::OK(), result->status());
    for(int i = 0; i < 4; i++)
        ASSERT_EQ(assertion[i],resultArr(i));
    // auto z = result->at(0);
    //ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));


}



TEST_F(DeclarableOpsTests7, TEST_WHERE_SCALAR) {
    std::vector<double> data;
    std::vector<double> mask;
    std::vector<double> put;
    std::vector<double> resultData;
    std::vector<double> assertion;
    for(Nd4jIndex i = 0; i < 4; i++) {
        data.push_back(i);
        if(i >  1) {
            assertion.push_back(5.0);
            mask.push_back(1);
        }
        else {
            assertion.push_back(i);
            mask.push_back(0);
        }

        resultData.push_back(0.0);
    }


    put.push_back(5.0);


    NDArray<double> x('c',{1,4},data);
    NDArray<double> maskArr('c',{1,4},mask);
    NDArray<double> putArr('c',{1,1},put);
    NDArray<double> resultArr('c',{1,4},resultData);
    nd4j::ops::where<double> op;
    //greater than test
    //            Nd4jStatus execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);

    auto result = op.execute({&maskArr,&x,&putArr},{&resultArr}, {},{3},false);
    // ASSERT_EQ(Status::OK(), result->status());
    for(int i = 0; i < 4; i++)
        ASSERT_EQ(assertion[i],resultArr(i));
    // auto z = result->at(0);
    //ASSERT_EQ(4,z->lengthOf());
    //ASSERT_TRUE(exp.isSameShape(z));


}