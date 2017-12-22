//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/batched_gemm_strided.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void _bgemms(NDArray<T> *vA, NDArray<T> *vB, NDArray<T> *vC, NDArray<T> *alphas, NDArray<T> *betas) {
                int P = vA->sizeAt(0);

                int M = vA->sizeAt(1);
                int N = vB->sizeAt(2);
                int K = vB->sizeAt(1);

                int ldA = 0;
                int ldB = 0;
                int ldC = 0;

                CBLAS_TRANSPOSE tA = (CBLAS_TRANSPOSE) 111;
                CBLAS_TRANSPOSE tB = (CBLAS_TRANSPOSE) 111;


                #pragma omp parallel for                   
                for (int p = 0; p < P; ++p) {
                    auto A = vA->buffer() + (vA->sizeAt(1) * vA->sizeAt(2));
                    auto B = vB->buffer() + (vB->sizeAt(1) * vB->sizeAt(2));
                    auto C = vC->buffer() + (vC->sizeAt(1) * vC->sizeAt(2));
                    auto alpha = alphas->getScalar(p);
                    auto beta = betas->getScalar(p);
                    for (int m = 0; m < M; ++m) {
                        for (int n = 0; n < N; ++n) {
                            T c_mnp = 0;

                            #pragma omp simd
                            for (int k = 0; k < K; ++k)
                                c_mnp += A[tA == CblasNoTrans ? (m + k * ldA) : (m * ldA + k)] * B[tB == CblasNoTrans ? (k + n * ldB) : (k * ldB + n)];

                            C[m + n * ldC] = alpha * c_mnp + beta * C[m + n * ldC];
                        } 
                    } 
                }
            };

            template void _bgemms<float>(NDArray<float> *A, NDArray<float> *B, NDArray<float> *C, NDArray<float> *alphas, NDArray<float> *betas);
            template void _bgemms<float16>(NDArray<float16> *A, NDArray<float16> *B, NDArray<float16> *C, NDArray<float16> *alphas, NDArray<float16> *betas);
            template void _bgemms<double>(NDArray<double> *A, NDArray<double> *B, NDArray<double> *C, NDArray<double> *alphas, NDArray<double> *betas);
        }
    }
}