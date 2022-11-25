using Deeplearning.Core.Attributes;
using System;

namespace Deeplearning.Core.Math.Linear
{

    /// <summary>
    /// 正交化
    /// </summary>
    public class Orthogonalization
    {


        public static Func<Matrix, QREventArgs> Decompose(object decType)
        {

            int.TryParse(decType.ToString(), out int type);

            Func<Matrix, QREventArgs> function = Householder;

            switch (type)
            {
                case 0:
                    function = CGS;
                    break;
                case 1:
                    function = MGS;
                    break;
                case 2:
                default:
                    function = Householder;
                    break;
            }


            return function;
        }

        public static Matrix HouseholderMatrix(Vector vector)
        {
            Vector w = Vector.Standardized(vector);

            Matrix I = MatrixFactory.UnitMatrix(w.Length);

            return I - 2 * w * w.T;
        }


        public static QREventArgs Householder(Matrix source)
        {

            int size = (int)MathF.Min(source.Row, source.Column);

            Matrix Q = MatrixFactory.UnitMatrix(size);

            Matrix R = source;

            int count = source.Column - 1;

            for (int i = 0; i < count; i++)
            {
                Matrix subMatrix = Matrix.Clip(R,i, i, R.Row, R.Row);

                Vector x = subMatrix.GetVector(0);

                Vector y = new Vector(x.Length);

                y[0] = Vector.Norm(x); //- MathF.Sign((float)x[0]) * (float) x.Norm(2);

                Vector z = x - y;

                Matrix temp = HouseholderMatrix(z);

                Matrix h = MatrixFactory.UnitMatrix(source.Row);

                h = Matrix.Replace(h,temp, i, i);

                R = h * R;
                /*
     *人工校正 理论上与household 矩阵相乘的向量除了第一个元素  其余都会是0
     * 当时程序上而已 浮点型 本身就是不稳定的 所以需要人工进行校正
     */
                for (int j = i + 1; j < source.Row; j++)
                {
                    R[j, i] = 0;
                }
                Q = Q * h;
            }



            //  R =  UpperTriangularMatrixCalibration(R);

            return new QREventArgs(Q, R);
        }
      
        /// <summary>
        /// 上三角矩阵校正，
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        private static Matrix UpperTriangularMatrixCalibration(Matrix source)
        {
            for (int c = 0; c < source.Column; c++)
            {
                for (int r = source.Row - 1; r > c; r--)
                {
                    source[r, c] = 0;
                }
            }
            return source;
        }

        [Completion(false)]
        public static QREventArgs Givens(Matrix source)
        {


            return new QREventArgs();

        }

        /// <summary>
        /// 传统格拉姆 斯密特正交
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static QREventArgs CGS(Matrix source)
        {

            Matrix matrix = source;

            Matrix Q = source;

            Matrix R = new Matrix(source.Column, source.Column);

            for (int i = 0; i < matrix.Column; i++)
            {
                Vector a = matrix.GetVector(i);

                Vector e = Vector.Standardized(a);

                for (int j = i - 1; j >= 0; j--)
                {
                    e = Q.GetVector(j);

                    double temp = a.T * e;

                    R[j, i] = (float)temp;

                    a = a - temp * e;

                    e = Vector.Standardized(a);
                }

                R[i, i] = Vector.Norm(a);// (float)a.Norm(2);
                Q = Matrix.Replace(Q,e, i);
            }

            R = UpperTriangularMatrixCalibration(R);

            return new QREventArgs(Q, R);
        }


        /// <summary>
        /// 改良Modified Gram-Schmidt
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static QREventArgs MGS(Matrix source)
        {
            Matrix matrix = source;

            int vectorCount = source.Column;

            Matrix Q = new Matrix(source.Row, source.Column);

            Matrix R = new Matrix(vectorCount, vectorCount);

            for (int i = 0; i < vectorCount; i++)
            {
                Vector b = matrix.GetVector(i);

                Vector e = Vector.Standardized(b);

                Q = Matrix.Replace(Q,e, i);

                R[i, i] = Vector.Norm(b);

                for (int j = i + 1; j < vectorCount; j++)
                {
                    b = matrix.GetVector(j);

                    double temp = b.T * e;

                    R[i, j] = temp;

                    b = b - temp * e;

                    matrix = Matrix.Replace(matrix, b, j);
                }
            }

            R = UpperTriangularMatrixCalibration(R);

            return new QREventArgs(Q, R);
        }

    }
}
