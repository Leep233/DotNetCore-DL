using Deeplearning.Core.Attributes;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.LinearAlgebra
{

    /// <summary>
    /// 正交化
    /// </summary>
    public class Orthogonalization
    {

        public static QRResult Householder(Matrix source)
        {

            Matrix Q = null;

            Matrix R = (Matrix)source.Clone();

            for (int i = 0; i < source.Columns - 1; i++)
            {
                // Matrix matrix = source.Clip(i, i, source.Rows, source.Columns);

                Matrix matrix = source.Clip(i, i, source.Rows, source.Rows);

                Vector x = matrix.GetVector(0);

                double norm = x.Norm(2);

                Vector y = new Vector(x.Length);

                y[0] = Validator.ZeroValidation(norm);

                Vector z = x - y;

                double m = z.Norm(2);

                Vector w = z / m;

                Matrix I = Matrix.UnitMatrix(x.Length);

                Matrix temp = I - 2 * w * w.T;

                if (i==0)
                {
                    Q= temp;

                    R = Q * R;
                }
                else
                {

                    Matrix h = Matrix.UnitMatrix(source.Rows);

                   // Matrix h = Matrix.DiagonalMatrix(1,source.Rows,source.Columns);

                    h = h.Replace(temp, i, i, temp.Rows, temp.Columns);

                    Q = h * Q;

                    R = h * R;
                }
            }
            Q = Q.T;

            return new  QRResult(Q, R);
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

            Matrix R = new Matrix(source.Columns, source.Columns);

            for (int i = 0; i < matrix.Columns; i++)
            {
                Vector a = matrix.GetVector(i);
                Vector e = Vector.Normalize(a);

                for (int j = i - 1; j >= 0; j--)
                {
                    e = Q.GetVector(j);
                    double temp = a.T * e;
                    R[j, i] = Validator.ZeroValidation(temp);
                    a = a - temp * e;
                    e = Vector.Normalize(a);
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

            int vectorCount = source.Columns;

            Matrix Q = new Matrix(source.Rows, source.Columns);

            Matrix R = new Matrix(vectorCount, vectorCount);

            for (int i = 0; i < vectorCount; i++)
            {
                Vector b = matrix.GetVector(i);

                Vector e = Vector.Normalize(b);

                Q = Q.Replace(e, i);

                R[i, i] = b.Norm(2);

                for (int j = i+1; j < vectorCount; j++)
                {
                    b = matrix.GetVector(j);

                    double temp = b.T * e;

                    R[i, j] = Validator.ZeroValidation(temp);

                    b = b - temp * e;

                    matrix.Replace(b, j);
                }              
            }
            return new QRResult(Q, R);
        }

    }
}
