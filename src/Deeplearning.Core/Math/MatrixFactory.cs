using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public static class MatrixFactory
    {

        public static Matrix Random(int row, int col)
        {
            double[,] scalers = new double[row, col];

            Random r = new Random(DateTime.Now.GetHashCode());

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    scalers[i, j] = r.NextDouble();
                }
            }
            return new Matrix(scalers);
        }

        /// <summary>
        /// 单位矩阵
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public static Matrix UnitMatrix(int size)
        {
            Matrix matrix = new Matrix(size, size);
            for (int i = 0; i < size; i++)
            {
                matrix[i, i] = 1;
            }
            return matrix;
        }
        /// <summary>
        /// 对角矩阵
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public static Matrix DiagonalMatrix(int row, int col, params float[] scalars)
        {
            Matrix matrix = new Matrix(row, col);

            int size = (int)MathF.Min(row, col);

            size = (int)MathF.Min(size, scalars.Length);

            for (int i = 0; i < size; i++)
            {
                matrix[i, i] = scalars[i];
            }

            return matrix;
        }

        public static Matrix DiagonalMatrix(int size, params double[] scalars)
        {
            Matrix matrix = new Matrix(size, size);

            size = (int)MathF.Min(size, scalars.Length);

            for (int i = 0; i < size; i++)
            {
                matrix[i, i] = scalars[i];
            }

            return matrix;
        }
        
        public static Matrix DiagonalMatrix(params double[] array)
        {
            int size = array.Length;

            Matrix matrix = new Matrix(size, size);

            for (int i = 0; i < size; i++)
            {
                matrix[i, i] = array[i];
            }

            return matrix;
        }
      
        public static Matrix DiagonalMatrix(Vector vector)
        {
            int size = vector.Length;

            Matrix matrix = new Matrix(size, size);

            for (int i = 0; i < size; i++)
            {
                matrix[i, i] = vector[i];
            }

            return matrix;
        }

    }
}
