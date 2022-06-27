﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math
{
    public static class Matrix
    {
        public const float MinValue = 0.0001f;

        public static float[,] Transpose(float[,] matrix)
        {

            int row = matrix.GetLength(0);
            int col = matrix.GetLength(1);

            float[,] result = new float[col, row];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }
            return result;
        }

        public static float[,] Add(float[,] matrix, float scalar)
        {
            return Add(matrix, scalar);
        }

        public static float[,] Add(float scalar, float[,] matrix)
        {
            int row = matrix.GetLength(0);
            int col = matrix.GetLength(1);

            float[,] result = new float[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = scalar + matrix[i, j];
                }
            }
            return result;
        }

        public static float[,] Add(float[,] matrix, float[] vector)
        {
            return Add(vector, matrix);
        }

        public static float[,] Add(float[] vector, float[,] matrix)
        {
            int col = matrix.GetLength(1);

            if (col > vector.Length) throw new ArgumentException("vector's length < matrix's column");

            int row = matrix.GetLength(0);

            float[,] result = new float[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = vector[j] + matrix[i, j];
                }
            }
            return result;
        }

        public static float[,] Add(float[,] matrixA, float[,] matrixB)
        {
            int row = matrixA.GetLength(0);
            int col = matrixA.GetLength(1);

            if (row != matrixB.GetLength(0) || col != matrixB.GetLength(1))
                throw new ArgumentException("matrix's size must be consistent");

            float[,] result = new float[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = matrixA[i, j] + matrixB[i, j];
                }
            }
            return result;
        }

        public static float[,] HadamardProduct(float[,] matrixA, float[,] matrixB)
        {
            int row = matrixA.GetLength(0);
            int col = matrixA.GetLength(1);

            if (row != matrixB.GetLength(0) || col != matrixB.GetLength(1))
                throw new ArgumentException("matrix's size must be consistent");

            float[,] result = new float[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = matrixA[i, j] * matrixB[i, j];
                }
            }
            return result;
        }

        public static float[,] Dot(float[,] matrixA, float[,] matrixB)
        {
            int size = matrixA.GetLength(1);
            if (size != matrixB.GetLength(0)) throw new ArgumentException("MatrixA's Column != MatrixB's Row");

            int row = matrixA.GetLength(0);
            int col = matrixB.GetLength(1);

            float sum = 0f;

            float[,] result = new float[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    for (int k = 0; k < size; k++)
                    {
                        sum += matrixA[i, k] * matrixB[k, j];
                    }
                    result[i, j] = sum;
                    sum = 0;
                }
            }
            return result;
        }

        public static float[,] Dot(float[,] matrix, float[] vector)
        {
            int col = vector.Length;

            if (col <= matrix.GetLength(1)) throw new ArgumentException("matrix's column  must Greater than or equal to vector's length");

            int row = matrix.GetLength(0);

            float[,] result = new float[row, 1];

            float sum = 0;

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    sum += matrix[i, j] * vector[j];
                }
                result[i, 0] = sum;
                sum = 0;
            }
            return result;
        }

        public static float Dot(float[] vectorA, float[] vectorB)
        {
            int length = vectorA.Length;
            if (length != vectorB.Length) throw new ArgumentException("vector's size must be consistent");

            float sum = 0;

            for (int i = 0; i < length; i++)
            {

                sum += vectorA[i] * vectorB[i];
            }
            return sum;
        }

        public static float Norm(float[] vector, float p)
        {

            float temp = 0;

            for (int i = 0; i < vector.Length; i++)
            {
                temp += MathF.Pow(MathF.Abs(vector[i]), p);
            }
            return temp;
        }

        public static float MaxNorm(float[] vector)
        {

            float result = MathF.Abs(vector[0]);

            for (int i = 1; i < vector.Length; i++)
            {
                float temp = vector[i];

                if (MathF.Abs(temp) > result)
                    result = temp;
            }

            return result;
        }
        public static float FrobeniusNorm(float[,] matrix)
        {
            float temp = 0;

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    temp += MathF.Pow(matrix[i, j], 2);
                }
            }
            return MathF.Sqrt(temp);
        }

        public static float Track(float[,] matrix)
        {
            int row = matrix.GetLength(0);

            int col = matrix.GetLength(1);

            int count = row > col ? col : row;

            float temp = 0;

            for (int i = 0; i < count; i++)
            {
                temp += matrix[i, i];
            }
            return temp;
        }

        //public static double FrobeniusNorm(float[,] matrix)
        //{        
        //   float temp = Tr(Dot(matrix, Transpose(matrix)));

        //    return MathF.Sqrt((float)temp);
        //}

        public static float[,] DiagonalMatrix(float[] vector)
        {

            int length = vector.Length;

            float[,] result = new float[length, length];

            for (int i = 0; i < length; i++)
            {
                result[i, i] = vector[i];
            }

            return result;
        }

        public static float Det(float[,] matrix)
        {

            return 0;

        }

        public static string Print(float[,] matrix, string format = "f2")
        {

            if (matrix is null) return "null";

            double row = matrix.GetLength(0);
            double col = matrix.GetLength(1);

            StringBuilder stringBuilder = new StringBuilder();

            for (int i = 0; i < col; i++)
            {
                stringBuilder.Append("[");

                for (int j = 0; j < row; j++)
                {
                    stringBuilder.Append($" {matrix[j, i].ToString(format)} ");
                }
                stringBuilder.Append("]\n");
            }

            return stringBuilder.ToString();
        }


    }
}