using Deeplearning.Core.Math.Common;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

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
            int sR = (int)MathF.Max(startRow, 0); //startRow < 0 ? 0 : startRow;
            int sC = (int)MathF.Max(startColumn,0); //startColumn < 0 ? 0 : startColumn;

            int eR = (int)MathF.Min(endRow, source.Row); //endRow <= source.Row ? endRow : source.Row;
            int eC = (int)MathF.Min(endColumn, source.Column); //endColumn <= source.Column ? endColumn : source.Column;

            int rSize = eR - sR;
            int cSize = eC - sC;

            Matrix matrix = new Matrix(rSize, cSize);

            Parallel.For(0, rSize, r => {
                for (int c = 0; c < cSize; c++)
                {
                    matrix[r, c] = source[r + sR, c + sC];
                }
            });

            //for (int r = 0; r < rSize; r++)
            //{
            //    for (int c = 0; c < cSize; c++)
            //    {
            //        matrix[r, c] = source[r + sR, c + sC];
            //    }
            //}
            return matrix;
        }

        public static Matrix Replace(this Matrix source,Matrix matrix,int offsetR,int offsetC,int rCount,int cCount)
        {
            if(offsetR>= source.Row || offsetC>= source.Column) throw new IndexOutOfRangeException();

            Parallel.For(0, matrix.Row, r => {
                for (int c = 0; c < matrix.Column; c++)
                {
                    source[r + offsetR, c + offsetC] = matrix[r, c];
                }
            });

            //for (int r = 0; r < matrix.Row; r++)
            //{
            //    for (int c = 0; c < matrix.Column; c++)
            //    {
            //        source[r + offsetR, c + offsetC] = matrix[r, c];
            //    }
            //}



            return source;

        }

  

        /// <summary>
        /// 中心化（零均值化）：是指变量减去它的均值。其实就是一个平移的过程，平移后所有数据的中心是（0，0）
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static (Matrix matrix, double[]avgs) Centralized(this Matrix source) 
        {
            int vectorCount = source.Column;

            int vectorLength = source.Row;

            Matrix matrix = new Matrix(vectorLength, vectorCount);

            double[] avgs = Average(source,AxisDirection.Horizontal);      

            for (int r = 0; r < vectorLength; r++)
            {
                double sum = 0;
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
                    matrix[r,c] = source[r, c] - avgs[r];
                }          
            }

            return (matrix,avgs);

        }

        public static Matrix Replace(this Matrix source, Vector vector, int colIndex) 
        {
            Parallel.For(0, vector.Length, i => { source[i, colIndex] = vector[i]; });


            return source;
        }


        public static double[] StandardDeviation(this Matrix source, AxisDirection orientation = AxisDirection.Horizontal) {
          
            int row = source.Row;
           
            int col = source.Column;

            double[] avgs = Average(source, orientation);

            double[] sdevs = null;

            switch (orientation)
            {
                case AxisDirection.Vertical:
                    {
                        sdevs = new double[col];

                        int n = row - 1;

                        Parallel.For(0, col, c => {

                            double sum = 0;

                            double avg = avgs[c];

                            for (int r = 0; r < row; r++)
                            {
                                double value = source[r, c];
                                sum += MathF.Pow((float)(value - avg), 2);
                            }

                            sdevs[c] = MathF.Sqrt((float)(sum / n));
                        });

                        //for (int c = 0; c < col; c++)
                        //{
                        //    double sum = 0;

                        //    double avg = avgs[c];

                        //    for (int r = 0; r < row; r++)
                        //    {
                        //        double value = source[r, c];
                        //        sum += MathF.Pow((float)(value - avg), 2);
                        //    }

                        //    sdevs[c] =MathF.Sqrt((float)(sum / n));
                        //}

                    }
                    break;
                case AxisDirection.Horizontal:
                    {
                        sdevs = new double[row];

                        int n = row - 1;

                        Parallel.For(0, row, r => {
                            double sum = 0;

                            double avg = avgs[r];

                            for (int c = 0; c < col; c++)
                            {
                                double value = source[r, c];
                                sum += MathF.Pow((float)(value - avg), 2);
                            }
                            sdevs[r] = MathF.Sqrt((float)sum / n);
                        });
                    }
                    break;
            }
            return sdevs;
        }

        /// <summary>
        /// 求方差
        /// </summary>
        /// <param name="source"></param>
        /// <param name="orientation"></param>
        /// <returns></returns>
        public static double[] Var(this Matrix source, AxisDirection orientation = AxisDirection.Horizontal) 
        {
            int row = source.Row;

            int col=  source.Column;

            double[] avgs = Average(source,orientation);

            double[] vars = null;           

            switch (orientation)
            {
                case AxisDirection.Vertical:
                    {
                        vars = new double[col];

                        int n = row - 1;

                        for (int c = 0; c < col; c++)
                          
                        {
                            float sum = 0;

                            double avg = avgs[c];

                            for (int r = 0; r < row; r++)
                            {
                                double value = source[r, c];
                                sum += MathF.Pow((float)(value - avg), 2);
                            }

                            vars[c] = sum / n;
                        }

                    }
                    break;
                case AxisDirection.Horizontal:
                    {
                        vars = new double[row];

                        int n = row - 1;

                        for (int r = 0; r < row; r++)                           
                        {
                            double sum = 0;

                            double avg = avgs[r];

                            for (int c = 0; c < col; c++)
                            {
                                double value = source[r, c];
                                sum += MathF.Pow((float)(value - avg), 2);
                            }
                            vars[r] = sum / n;
                        }
                    }
                    break;
            }
            return vars;          
        }

        /// <summary>
        /// 求均值
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static double[] Average(this Matrix source, AxisDirection  orientation = AxisDirection.Horizontal) {

            int vectorCount = source.Column;

            int vectorLength = source.Row;

            double[] avgs = null;

            switch (orientation)
            {
                case AxisDirection.Vertical:
                    {                     
                            avgs = new double[vectorCount];
                            //求每一列均值
                            for (int c = 0; c < vectorCount; c++)                               
                            {
                            double sum = 0;
                                for (int r = 0; r < vectorLength; r++)
                                {
                                    sum += source[r, c];
                                }
                                avgs[c] = sum / vectorLength;
                            }                     
                    }
                    break;
                case AxisDirection.Horizontal:
                    {
                       avgs = new double[vectorLength];
                        //求每一行均值
                        for (int r = 0; r < vectorLength; r++)
                        {
                            double sum = 0;
                            for (int c = 0; c < vectorCount; c++)
                            {
                                sum += source[r, c];
                            }
                            avgs[r] = sum / vectorCount;
                        }                     
                    }
                    break;
            }

            return avgs;
        }

        /// <summary>
        /// 矩阵归一化
        /// </summary>
        /// <param name="source"></param>
        /// <param name="mode">0，将所有列向量归一，其他将行向量归一</param>
        /// <returns></returns>
        public static Matrix Normalized(this Matrix source) 
        {
            int vectorCount = source.Column;

            int vectorLength = source.Row;

            Matrix matrix = new Matrix(vectorLength, vectorCount);

            double[] avgs = new double[vectorLength];

            //求每一行均值
            for (int r = 0; r < vectorLength; r++)
            {
                double sum = 0;     
                for (int c = 0; c < vectorCount; c++)
                {
                    sum += source[r, c];
                }
                avgs[r] = sum / vectorCount;
            }
            //求每一行标准差

            double[] sdevs = new double[vectorLength];

            for (int r = 0; r < vectorLength; r++)
            {
                double sum = 0;
                double avg = avgs[r];
                for (int c = 0; c < vectorCount; c++)
                {
                    double value = source[r, c];
                    sum += MathF.Pow((float)(value - avg), 2);
                }
                sdevs[r] = MathF.Sqrt((float)(sum / vectorCount));
            }

            // （x-均值）/标准差
            for (int r = 0; r < vectorLength; r++)
            {
                for (int c = 0; c < vectorCount; c++)
                {
                    matrix[r, c] = (source[r, c] - avgs[r])/ (sdevs[r]);
                }
            }

            return matrix;
        }

        /// <summary>
        /// 获取主对角所有元素
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static double[] DiagonalElements(this Matrix source) 
        {
    
            int r=(int) MathF.Min(source.Row,source.Column);

            double[] vector = new double[r];

            for (int i = 0; i < r; i++)
            {
                vector[i] = source[i, i];
            }
            return vector;
        }

        public static Matrix Var(this Matrix source)
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

        public static Matrix Cov(this Matrix source) {

            float n = source.Column -1;//;//

            Matrix m1 = source;
            Matrix m2 = source.T;

            int rows = m1.Row;

            int same = m1.Column;

            int cols = m2.Column;

            Matrix result = new Matrix(m1.Row, m2.Column);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double temp = 0;

                    for (int k = 0; k < same; k++)
                    {
                        temp += m1[i, k] * m2[k, j];
                    }
                    result[i, j] = temp/n;
                }
            }
            return result;

           // return (source * source.T) / n;
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
                double value = D[i, i];
                if (value == 0) continue;
                D_pInv[i, i] = 1 / value;
            }

            return V * D_pInv * U.T;
        }
    }
}
