using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace Deeplearning.Core.Math
{
    /// <summary>
    /// 特征分解测试 
    /// </summary>
    public class EigTest
    {

        public static  float Eig(Matrix source) 
        {
            if (!source.IsSquare) throw new Exception("矩阵必须是方阵");

            int size = source.Rows;

            int step = 1000;

            double e = 10E-8;

            float lr = 0.0001f;

            float λ = 999;

            for (int i = 0; i < step; i++)
            {
                Matrix m = source - Matrix.DiagonalMatrix(λ, size);

                float k = (m).det;

                if (MathF.Abs(k) == e)
                    break;

                λ -= lr * k;
            }
            return λ;
        }


        public static float Eigenvalue(Matrix source,Vector eigenVector) 
        {
            Vector sv = Vector.Normalize(eigenVector);

            float value = sv.T * source * sv;            

            return value;
        }

    }
}
