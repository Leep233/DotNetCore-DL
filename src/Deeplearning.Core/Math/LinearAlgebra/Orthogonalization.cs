using Deeplearning.Core.Attributes;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math.LinearAlgebra
{

    /// <summary>
    /// 正交化
    /// </summary>
    public class Orthogonalization
    {

        public static Matrix HouseholderMatrix(Vector vector) 
        {
            Vector w = Vector.Standardized(vector);

            Matrix I = Matrix.UnitMatrix(w.Length);

            return I - 2 * w * w.T;
        }


        public static QRResult Householder(Matrix source)
        {

            int size = (int)MathF.Min(source.Row, source.Column);

            Matrix Q = Matrix.UnitMatrix(size);

            Matrix R = (Matrix)source.Clone();

            int vectorCounts = source.Column - 1;

            for (int i = 0; i < source.Column - 1; i++)
            {
                Matrix subMatrix = R.Clip(i, i, R.Row, R.Row);

                Vector x = subMatrix.GetVector(0);

                Vector y = new Vector(x.Length);

                y[0] = -MathF.Sign((float)x[0]) * x.Norm(2);

                Vector z = y - x;

                Matrix temp = HouseholderMatrix(z);

                Matrix h = Matrix.UnitMatrix(source.Row);

                h = h.Replace(temp, i, i, temp.Row, temp.Column);

                Matrix a_1 = h * R;
                R = a_1;
                Q = Q * h;
            }

            /*

       */

            return new QRResult(Q, R);
        }

      

        [Completion(false)]
        public static QRResult Givens(Matrix source)
        {
            Matrix Q = null;
            Matrix R = null;

            return new QRResult();

        }

        /// <summary>
        /// 传统格拉姆 斯密特正交
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static QRResult CGS(Matrix source)
        {

            Matrix matrix = (Matrix)source.Clone();

            Matrix Q = (Matrix)source.Clone();

            Matrix R = new Matrix(source.Column, source.Column);

            for (int i = 0; i < matrix.Column; i++)
            {
                Vector a = matrix.GetVector(i);
                Vector e = Vector.Standardized(a);

                for (int j = i - 1; j >= 0; j--)
                {
                    e = Q.GetVector(j);
                    float temp = a.T * e;
                    R[j, i] = temp;
                    a = a - temp * e;
                    e = Vector.Standardized(a);
                }

                R[i, i] = a.Norm(2);
                Q = Q.Replace(e, i);
            }
            return new QRResult(Q, R);
        }


        /// <summary>
        /// 改良Modified Gram-Schmidt
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static QRResult MGS(Matrix source)
        {
            Matrix matrix = (Matrix)source.Clone();

            int vectorCount = source.Column;

            Matrix Q = new Matrix(source.Row, source.Column);

            Matrix R = new Matrix(vectorCount, vectorCount);

            for (int i = 0; i < vectorCount; i++)
            {
                Vector b = matrix.GetVector(i);

                Vector e = Vector.Standardized(b);

                Q = Q.Replace(e, i);

                R[i, i] = b.Norm(2);

                for (int j = i+1; j < vectorCount; j++)
                {
                    b = matrix.GetVector(j);

                    float temp = b.T * e;

                    R[i, j] = temp;

                    b = b - temp * e;

                    matrix.Replace(b, j);
                }              
            }
            return new QRResult(Q, R);
        }

    }
}
