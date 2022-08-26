using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public static class MatrixExtension
    {
        public static Matrix Clip(this Matrix source, int startRow, int startColumn, int endRow, int endColumn)
        {
            int sR = startRow < 0 ? 0 : startRow;
            int sC = startColumn < 0 ? 0 : startColumn;            

            int eR = endRow <= source.Rows ? endRow : source.Rows;
            int eC = endColumn <= source.Columns ? endColumn : source.Columns;

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
            if(offsetR>= source.Rows || offsetC>= source.Columns) throw new IndexOutOfRangeException();

            for (int r = 0; r < matrix.Rows; r++)
            {
                for (int c = 0; c < matrix.Columns; c++)
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

            int rowCount = source.Rows;

            int colCount = source.Columns;

            switch (mode)
            {
                case 0:
                    {
                        for (int c = 0; c < colCount; c++)
                        {
                            double sum = 0;

                            for (int r = 0; r < rowCount; r++)
                            {
                                sum += MathF.Pow((float)source[r, c], 2);
                            }                       

                            double norm = Validator.ZeroValidation(MathF.Sqrt((float)sum));

                            for (int r = 0; r < rowCount; r++)
                            {
                                double value = source[r, c];

                                source[r, c] = norm == 0 ? 0 : value / norm;
                            }
                        }
                    }
                    break;
                default:
                    {
                        for (int r = 0; r < rowCount; r++)
                           
                        {
                            double sum = 0;

                            for (int c = 0; c < colCount; c++)
                            {
                                sum += MathF.Pow((float)source[r, c], 2);
                            }

                            double norm = Validator.ZeroValidation(MathF.Sqrt((float)sum));

                            for (int c = 0; c < colCount; c++)
                            {
                                double value = source[r, c];
                                source[r, c] = norm == 0 ? 0 : value / norm;
                            }
                        }
                    }
                    break;
            }

            return source;
        }
    }
}
