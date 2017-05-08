//
// @author raver119@gmail.com
//

#ifndef LIBND4J_AGGREGATES_H
#define LIBND4J_AGGREGATES_H

#include <ops/aggregate_ops.h>
#include <helpers/helper_ptrmap.h>

#define AGGREGATE_OPS \
        (0, aggregateOps::HierarchicSoftmax) ,\
        (1, aggregateOps::Dot) ,\
        (2, aggregateOps::Axpy) ,\
        (3, aggregateOps::SkipGram) ,\
        (4, aggregateOps::CBOW) ,\
        (5, aggregateOps::GEMM)


namespace functions {
    namespace aggregate {

        template<typename T>
        class AggregatedFunction {

        public:
#ifdef __CUDACC__
            template<typename OpClass>
            __device__ inline static void execCuda(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  T *realArguments, int numRealArguments) {
                OpClass::executeAggregateCuda(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
            }
#endif

            template<typename OpClass>
            inline static void exec(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  T *realArguments, int numRealArguments) {
                OpClass::executeAggregate(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
            }

            inline static void exec(int opNum, T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
                DISPATCH_BY_OPNUM(exec, PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), AGGREGATE_OPS);
            }


            template <typename OpClass>
            inline static void aggregateBatchReduceGeneric(int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments) {
                // probably, we don't want too much threads as usually
                int _threads = nd4j::math::nd4j_min<int>(numAggregates, omp_get_max_threads());

                nd4j::PointersHelper<T> helper(ptrToArguments,
                                               numAggregates,
                                               maxArgs,
                                               maxShapes,
                                               maxIntArrays,
                                               maxIntArraySize,
                                               maxIdx,
                                               maxReals);

                // special case here, we prefer spread arrangement here, all threads are detached from each other
#pragma omp parallel for num_threads(_threads) schedule(guided) proc_bind(spread) default(shared)
                for (int i = 0; i < numAggregates; i++) {
                    int **intArrays = new int *[maxIntArrays];

                    T **arguments = helper.getArguments(i);
                    int **shapes = helper.getShapeArguments(i);
                    int *idxArg = helper.getIndexArguments(i);
                    T *realArg = helper.getRealArguments(i);

                    for (int e = 0; e < maxIntArrays; e++) {
                        intArrays[e] = helper.getIntArrayArguments(i, e);
                    }

                    // TODO: call for Reduce function with PROPER params hehe
                    functions::reduce::ReduceFunction<T>::template execScalar<OpClass>(nullptr, nullptr, nullptr);

                    delete [] intArrays;
                }
            };
		};
    }
}




#ifdef __CUDACC__

template <typename T, typename OpClass>
__device__ void aggregateKernelGeneric(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
    functions::aggregate::AggregatedFunction<T>:: template execCuda<OpClass>(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
};


template <typename T, typename OpClass>
__device__ void aggregateBatchKernelGeneric(int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments) {

    nd4j::PointersHelper<T> helper(ptrToArguments, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

    // TODO: we probably should lift this restriction
    __shared__ int *intArrays[32];

    __shared__ T **arguments;
    __shared__ int **shapes;
    __shared__ int *idxArg;
    __shared__ T *realArg;

    for(int r = blockIdx.x; r < numAggregates; r += gridDim.x) {
        if (threadIdx.x == 0) {
            arguments = helper.getArguments(r);
            shapes = helper.getShapeArguments(r);
            idxArg = helper.getIndexArguments(r);
            realArg = helper.getRealArguments(r);
        }

        // we fill intArrays param in parallel within block
        if (threadIdx.x < 32 && threadIdx.x < maxIntArrays) {
            intArrays[threadIdx.x] = helper.getIntArrayArguments(r, threadIdx.x);
        }
        __syncthreads();

        functions::aggregate::AggregatedFunction<T>::template execCuda<OpClass>(arguments, helper.getNumArguments(r), shapes, helper.getNumShapeArguments(r), idxArg, helper.getNumIndexArguments(r), intArrays, helper.getNumIntArrayArguments(r), realArg, helper.getNumRealArguments(r));
    }
};

/**
*   This executor is suited for launching multiple reduce ops at once
**/
template <typename T, typename OpClass>
__device__ void aggregateBatchReduceKernelGeneric(int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments) {
    nd4j::PointersHelper<T> helper(ptrToArguments, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

    // TODO: we probably should lift this restriction
    __shared__ int *intArrays[32];

    __shared__ T **arguments;
    __shared__ int **shapes;
    __shared__ int *idxArg;
    __shared__ T *realArg;

    for(int r = blockIdx.x; r < numAggregates; r += gridDim.x) {
        if (threadIdx.x == 0) {
            arguments = helper.getArguments(r);
            shapes = helper.getShapeArguments(r);
            idxArg = helper.getIndexArguments(r);
            realArg = helper.getRealArguments(r);
        }

        // we fill intArrays param in parallel within block
        if (threadIdx.x < 32 && threadIdx.x < maxIntArrays) {
            intArrays[threadIdx.x] = helper.getIntArrayArguments(r, threadIdx.x);
        }
        __syncthreads();

        // TODO: here we'll issue call to special version of ReduceFunction.transformCuda, which will be single-block always
    }
};

// simple aggregates
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateKernelGeneric, float, INPUT(float **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateKernelGeneric, double, INPUT(double **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, double *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateKernelGeneric, float16, INPUT(float16 **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float16 *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))


// batched aggregates
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchKernelGeneric, float, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchKernelGeneric, float16, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchKernelGeneric, double, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(AGGREGATE_OPS))

DISPATCH_KERNEL_SIMPLE(aggregateBatchReduce_, aggregateBatchReduceKernelGeneric, float, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateBatchReduce_, aggregateBatchReduceKernelGeneric, float16, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateBatchReduce_, aggregateBatchReduceKernelGeneric, double, INPUT(int numAggregates, int ops, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments), PARAMS(numAggregates, ops, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals, ptrToArguments), OPS_A(REDUCE_OPS))



#endif

#endif //LIBND4J_AGGREGATES_H
