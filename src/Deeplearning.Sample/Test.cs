using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Deeplearning.Sample
{
    internal class Test
    {

   
        public static float Eig(Matrix A) {

            //这里我们只需要做一个极小值来判断就可以了
            float result = 0;
            float targetdet = 10E-5F;
            float learningRate = 10E-3F;
            //开始对函数进行梯度下降
            int loopCount = 1000;
            // λ 得知 其实就是求 0 ≈ det(A) 的梯度
            //1.随机出来一个 λ,这里注意我们的
            //|λ| =0 或者 = 1 就没有意义了
            //所以建议使用标准正态分布进行分布(注意 这里不适用分布函数也是可以的 但是需要判断|λ|=0.=1的情况)
            float λ = new Random().Next(-10,10);//ProbabilityDistribution.StandardNormalDistribution();

            for (int i = 0; i < loopCount; i++)
            {
                float det = Function(A, λ);

                if (MathF.Abs(det) <= targetdet)
                {
                    result = λ;

                    break;
                }

                λ -= det * learningRate;
            }
            return result;
        }

        private static float Function(Matrix A, float λ) 
        {
            int size = A.Rows;  
            //A - λI
            Matrix λI = Matrix.DiagonalMatrix(λ, size);

            for (int r = 0; r < size; r++)
            {
                for (int c = 0; c < size; c++)
                {
                    λI[r, c] -= A[r, c];
                }
            }

          return Matrix.Det(λI);
        }
    }
}
