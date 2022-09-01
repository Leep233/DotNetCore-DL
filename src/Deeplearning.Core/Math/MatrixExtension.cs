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

        /// <summary>
        /// 中心化（零均值化）：是指变量减去它的均值。其实就是一个平移的过程，平移后所有数据的中心是（0，0）
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static Matrix Centralized(this Matrix source) 
        {
            int vectorCount = source.Column;
            int vectorLength = source.Row;

            float[] avgs = new float[vectorLength];

            for (int r = 0; r < vectorLength; r++)
            {
                float sum = 0;
                for (int c = 0; c < vectorCount; c++)
                {
                    sum += source[r, c];
                }
                avgs[r] = sum / vectorCount;
            }

            for (int r = 0; r < vectorLength; r++)
            {
                for (int c = 0; c < vectorCount; c++)
                {
                     source[r, c] -= avgs[r];
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
        public static Matrix Normalized(this Matrix source) 
        {
            Matrix matrix = new Matrix(source.Row,source.Column);

            for (int c = 0; c < source.Column; c++)
            {
                float sum = 0;

                for (int r = 0; r < source.Row; r++)
                {
                    sum += source[r, c];
                }

                for (int r = 0; r < source.Row; r++)
                {
                    matrix[r, c] = sum == 0?0 : source[r, c] / sum;
                }
            }
            return matrix;
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
  
   

            float m = elementCount - 1;

        
            return (source.T * source) / m; ;
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
