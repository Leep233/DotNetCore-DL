using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Deeplearning.Core.Math
{
    public static class Linear
    {
       public static double[,] Transpose(double[,] matrix)
        {

            int row = matrix.GetLength(0);
            int col = matrix.GetLength(1);

            double[,] result = new double[col, row];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }
            return result;
        }

        public static double[,] Add(double[,] matrix, double scalar)
        {       
            return Add(matrix, scalar);
        }

        public static double[,] Add(double scalar, double[,] matrix)
        {
            int row = matrix.GetLength(0);
            int col = matrix.GetLength(1);
            double[,] result = new double[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = scalar + matrix[i, j];
                }
            }
            return result;
        }

        public static double[,] Add(double[,] matrix, double[] vector)
        {
            return Add(vector,matrix);
        }
        
        public static double[,] Add(double[] vector, double[,] matrix)
        {
            int col = matrix.GetLength(1);
            if (col > vector.Length) throw new ArgumentException("vector's length < matrix's column");

            int row = matrix.GetLength(0);
            double[,] result = new double[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = vector[j] + matrix[i, j];
                }
            }
            return result;
        }

        public static double[,] Add(double[,] matrixA, double[,] matrixB) 
        { 
            int row = matrixA.GetLength(0);
            int col = matrixA.GetLength(1);

            if (row != matrixB.GetLength(0) || col != matrixB.GetLength(1))
                throw new ArgumentException("matrix's size must be consistent");

            double[,] result = new double[row,col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = matrixA[i,j] + matrixB[i,j];
                }
            }
            return result;
        }

        public static double[,] HadamardProduct(double[,] matrixA, double[,] matrixB) 
        {
            int row = matrixA.GetLength(0);
            int col = matrixA.GetLength(1);

            if (row != matrixB.GetLength(0) || col != matrixB.GetLength(1))
                throw new ArgumentException("matrix's size must be consistent");

            double[,] result = new double[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = matrixA[i, j] * matrixB[i, j];
                }
            }
            return result;
        }

        public static double[,] Dot(double[,] matrixA, double[,] matrixB)
        {
            int size = matrixA.GetLength(1);
            if (size != matrixB.GetLength(0)) throw new ArgumentException("MatrixA's Column != MatrixB's Row");

            int row = matrixA.GetLength(0);
            int col = matrixB.GetLength(1);

            double sum = 0;

            double[,] result = new double[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    for (int k = 0; k < size; k++)
                    {
                        sum += matrixA[i, k] * matrixB[k, j];
                    }
                    result[i, j] = sum;
                    sum =0;
                }
            }
            return result;
        }

        public static double[,] Dot(double[,] matrix, double[] vector) 
        { 
            int col = vector.Length;

            if (col <= matrix.GetLength(1)) throw new ArgumentException("matrix's column  must Greater than or equal to vector's length");

            int row = matrix.GetLength(0);

            double [,] result = new double [row, 1];

            double sum = 0;

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

        public static double Dot(double [] vectorA,double [] vectorB) 
        { 
            int length = vectorA.Length;
            if (length != vectorB.Length) throw new ArgumentException("vector's size must be consistent");

            double sum = 0;
   
            for (int i = 0; i < length; i++) {

                sum += vectorA[i] * vectorB[i];
            }
            return sum;
        }

        public static double Norm(double [] vector,double p) {

            double temp = 0;

            for (int i = 0; i < vector.Length; i++)
            {
                temp += MathF.Pow(MathF.Abs((float)vector[i]), (float)p);
            }
            return temp;
        }

        public static double MaxNorm(double[] vector) {

            double result = MathF.Abs((float)vector[0]);

            for (int i = 1; i < vector.Length; i++)
            {
               float temp = (float)vector[i];

                if (MathF.Abs(temp) > result)
                    result = temp;
            }

            return result;
        }
        public static double FrobeniusNorm(double [,] matrix) 
        {
            float temp = 0;

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    temp += MathF.Pow((float)matrix[i,j], 2);
                }
            }
            return MathF.Sqrt(temp);
        }

        public static double Track(double[,] matrix)
        {
           int row = matrix.GetLength(0);
            
            int col = matrix.GetLength(1);

            int count = row > col? col : row;

            double temp = 0;

            for (int i = 0; i < count; i++)
            {
                temp += matrix[i, i];
            }
            return temp;
        }

        //public static double FrobeniusNorm(double[,] matrix)
        //{        
        //   double temp = Tr(Dot(matrix, Transpose(matrix)));

        //    return MathF.Sqrt((float)temp);
        //}

        public static double[,] DiagonalMatrix(double[] vector) {
            
            int length = vector.Length;

            double[,] result = new double[length, length];

            for (int i = 0; i < length; i++)
            {
                result[i, i] = vector[i];
            }

            return result;
        }

        public static double Det(double[,] matrix) {

            return 0;

        }


        public static string Print(double[,] matrix,string format = "f2")
        {

            if(matrix is null) return "null";

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
