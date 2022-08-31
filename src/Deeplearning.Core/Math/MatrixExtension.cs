using Deeplearning.Core.Math.LinearAlgebra;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Deeplearning.Core.Math
{
    public static class MatrixExtension
    {


        public static Vector[] Vectors(this Matrix source)
        {

            int row = source.Row;
            int col = source.Column;

            Vector[] vs = new Vector[col];

            for (int c = 0; c < col; c++)
            {
                vs[c] = new Vector(row);
                for (int r = 0; r < row; r++)
                {
                    vs[c][r] = source[r, c];
                }
            }

            return vs;
        }

        public static Vector GetVector(this Matrix source,int colIndex)
        {

            int row = source.Row;
            int col = source.Column;

            if (col <= colIndex) throw new ArgumentOutOfRangeException("越界");

            Vector vector = new Vector(row);

            for (int i = 0; i < row; i++)
            {
                vector[i] = source[i, colIndex];
            }
            return vector;
        }

        public static Matrix Clip(this Matrix source, int startRow, int startColumn, int endRow, int endColumn)
        {
            int sR = startRow < 0 ? 0 : startRow;
            int sC = startColumn < 0 ? 0 : startColumn;            

            int eR = endRow <= source.Row ? endRow : source.Row;
            int eC = endColumn <= source.Column ? endColumn : source.Column;

            int rSize = eR - sR;
            int cSize = eC - sC;

            Matrix matrix = new Matrix(rSize, cSize);

            for (int r = 0; r < rSize; r++)
            {
                for (int c = 0; c < cSize; c++)
                {
                    matrix[r, c] = source[r + sR, c + sC];
                }
            }
            return matrix;
        }

        public static Matrix Replace(this Matrix source,Matrix matrix,int offsetR,int offsetC,int rCount,int cCount)
        {
            if(offsetR>= source.Row || offsetC>= source.Column) throw new IndexOutOfRangeException();

            for (int r = 0; r < matrix.Row; r++)
            {
                for (int c = 0; c < matrix.Column; c++)
                {
                    source[r+ offsetR, c+ offsetC] = matrix[r, c];
                }
            }

            return source;

        }


        public static Matrix Replace(this Matrix source, Vector vector, int colIndex) 
        {
            for (int i = 0; i < vector.Length; i++)
            {
                source[i, colIndex] = vector[i];
            }

            return source;
        }

        /// <summary>
        /// 矩阵归一化
        /// </summary>
        /// <param name="source"></param>
        /// <param name="mode">0，将所有列向量归一，其他将行向量归一</param>
        /// <returns></returns>
        public static Matrix Normalize(this Matrix source, int mode = 0) 
        {

            int rowCount = source.Row;

            int colCount = source.Column;

            switch (mode)
            {
                case 0:
                    {
                        for (int c = 0; c < colCount; c++)
                        {
                            float sum = 0;

                            for (int r = 0; r < rowCount; r++)
                            {
                                sum += MathF.Pow((float)source[r, c], 2);
                            }

                            float norm = Validator.ZeroValidation(MathF.Sqrt((float)sum));

                            for (int r = 0; r < rowCount; r++)
                            {
                                float value = source[r, c];

                                source[r, c] = norm == 0 ? 0 : value / norm;
                            }
                        }
                    }
                    break;
                default:
                    {
                        for (int r = 0; r < rowCount; r++)
                           
                        {
                            float sum = 0;

                            for (int c = 0; c < colCount; c++)
                            {
                                sum += MathF.Pow((float)source[r, c], 2);
                            }

                            float norm = Validator.ZeroValidation(MathF.Sqrt((float)sum));

                            for (int c = 0; c < colCount; c++)
                            {
                                float value = source[r, c];
                                source[r, c] = norm == 0 ? 0 : value / norm;
                            }
                        }
                    }
                    break;
            }

            return source;
        }

        /// <summary>
        /// 获取主对角所有元素
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static float[] DiagonalVector(this Matrix source) 
        {
    
            int r=(int) MathF.Min(source.Row,source.Column);

            float[] vector = new float[r];

            for (int i = 0; i < r; i++)
            {
                vector[i] = source[i, i];
            }
            return vector;
        }

        public static Matrix VarianceMatrix(this Matrix source)
        {
            int vectorCount = source.Column;
            int elementCount = source.Row;

            Matrix matrix = new Matrix(elementCount, vectorCount);

            double sum = 0;

            for (int i = 0; i < vectorCount; i++)
            {
                sum = 0;

                for (int j = 0; j < elementCount; j++)
                {
                    sum += source[j, i];
                }
                for (int j = 0; j < elementCount; j++)
                {
                    double value = source[j, i];

                    value /= sum;

                    matrix[j,i] = MathF.Pow((float)value,2);
                }
            }
            return matrix;
        }

        public static Matrix CovarianceMatrix(this Matrix source) {
            int vectorCount = source.Column;
            int elementCount = source.Row;

            Matrix matrix = new Matrix(elementCount, vectorCount);
   

            float m = elementCount - 1;

            for (int i = 0; i < vectorCount; i++)
            {
                float sum = 0;

                for (int j = 0; j < elementCount; j++)
                {
                    sum += source[j, i];
                }
                for (int j = 0; j < elementCount; j++)
                {
                    float value = source[j, i];

                    value /= sum;

                    matrix[j, i] = MathF.Pow((float)value, 2)/ m;
                }
            }
            return matrix;
        }

        public static Matrix PInv(this Matrix source,Func<Matrix,QRResult> orthogonaliztionFunction=null) {

            if (orthogonaliztionFunction is null)
                orthogonaliztionFunction = Orthogonalization.Householder;

            SVDResult svdResult = MatrixDecomposition.SVD(source, orthogonaliztionFunction);      

            Matrix D = svdResult.S;
            Matrix V = svdResult.V;
            Matrix U = svdResult.U;

            Matrix D_pInv = new Matrix(D.Column,D.Row);

            int r = (int)MathF.Min(D.Row, D.Column);

            for (int i = 0; i < r; i++)
            {
                float value = D[i, i];
                if (value == 0) continue;
                D_pInv[i, i] = 1 / value;
            }

            return V * D_pInv * U.T;
        }
    }
}
