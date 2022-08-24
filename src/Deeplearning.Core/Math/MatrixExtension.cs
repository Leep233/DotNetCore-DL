using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public static class MatrixExtension
    {
        public static Matrix Clip(this Matrix soruce, int startRow, int startColumn, int endRow, int endColumn)
        {
            int sR = startRow < 0 ? 0 : startRow;
            int sC = startColumn < 0 ? 0 : startColumn;            

            int eR = endRow <= soruce.Rows ? endRow : soruce.Rows;
            int eC = endColumn <= soruce.Columns ? endColumn : soruce.Columns;

            int rSize = eR - sR;
            int cSize = eC - sC;

            Matrix matrix = new Matrix(rSize, cSize);

            for (int r = 0; r < rSize; r++)
            {
                for (int c = 0; c < cSize; c++)
                {
                    matrix[r, c] = soruce[r + sR, c + sC];
                }
            }
            return matrix;
        }

        public static Matrix Replace(this Matrix soruce,Matrix matrix,int offsetR,int offsetC,int rCount,int cCount)
        {
            if(offsetR>= soruce.Rows || offsetC>= soruce.Columns) throw new IndexOutOfRangeException();

            for (int r = 0; r < matrix.Rows; r++)
            {
                for (int c = 0; c < matrix.Columns; c++)
                {
                    soruce[r+ offsetR, c+ offsetC] = matrix[r, c];
                }
            }

            return soruce;

        }
    }
}
