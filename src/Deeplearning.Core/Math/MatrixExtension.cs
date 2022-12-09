using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public static class MatrixExtension
    {

        public const double MIN_VALUE = 10E-15;

        /// <summary>
        /// 求均值
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="axis">默认-1：所有均值；0：每列均值；1：每行均值</param>
        /// <returns></returns>
        public static Vector Mean(this Matrix matrix,int axis = -1) 
        {
            Vector vector;

            switch (axis)
            {
                case 0:
                    {
                        vector = new Vector(matrix.Column);

                        for (int i = 0; i < matrix.Column; i++)
                        {
                            double sum = 0;
                            for (int j = 0; j < matrix.Row; j++)
                            {
                                sum += matrix[j, i];
                            }
                            vector[i] = sum / matrix.Row;
                        }

                    }
                    break;
                case 1:
                    {
                        vector = new Vector(matrix.Row);

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            double sum = 0;
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += matrix[i, j];
                            }
                            vector[i] = sum / matrix.Column;
                        }

                    }
                    break;
                case -1:
                default:
                    { 
                        vector = new Vector(1);

                        int length = matrix.Row * matrix.Column;

                        double sum = 0;

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += matrix[i, j];
                            }
                        }

                        vector[0] = sum/length;                    
                    }
                    break;
            }

            return vector;
        }

        /// <summary>
        /// 方差
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="axis">默认-1：所有均值；0：每列均值；1：每行均值</param>
        /// <returns></returns>
        public static Vector Var(this Matrix matrix, int axis = -1) 
        {
            Vector vector;

            switch (axis)
            {
                case 0:
                    {
                        vector = new Vector(matrix.Column);

                        for (int i = 0; i < matrix.Column; i++)
                        {
                            double sum = 0;
                            for (int j = 0; j < matrix.Row; j++)
                            {
                                sum += matrix[j, i];
                            }
                            double avg = sum / matrix.Row;

                            sum = 0;

                            for (int j = 0; j < matrix.Row; j++)
                            {
                                sum += MathF.Pow((float)(matrix[j, i]-avg),2);
                            }
                            vector[i] = sum/matrix.Row;
                        }
                    }
                    break;
                case 1:
                    {
                        vector = new Vector(matrix.Row);

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            double sum = 0;
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += matrix[i, j];
                            }

                            double avg = sum / matrix.Column;

                            sum = 0;

                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += MathF.Pow((float)(matrix[i,j] - avg), 2);
                            }
                            vector[i] = sum / matrix.Column;
                        }
                    }
                    break;
                case -1:
                default:
                    {
                        vector = new Vector(1);

                        int length = matrix.Row * matrix.Column;

                        double sum = 0;

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += matrix[i, j];
                            }
                        }

                        double avg = sum / length;

                        sum = 0;

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += MathF.Pow((float)(matrix[i, j] - avg), 2);
                            }
                        }

                        vector[0] = sum / length;

                    }
                    break;
            }

            return vector;
        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="axis">默认-1：所有均值；0：每列均值；1：每行均值</param>
        /// <returns></returns>
        public static (Vector stds,Vector means) StandardDeviation(this Matrix matrix, int axis = -1)
        {
            Vector stds;

            Vector means;

            switch (axis)
            {
                case 0:
                    {
                        stds = new Vector(matrix.Column);

                        means = new Vector(matrix.Column);

                        for (int i = 0; i < matrix.Column; i++)
                        {
                            double sum = 0;
                            for (int j = 0; j < matrix.Row; j++)
                            {
                                sum += matrix[j, i];
                            }
                            double avg = sum / matrix.Row;

                            means[i] = avg;

                            sum = 0;

                            for (int j = 0; j < matrix.Row; j++)
                            {
                                sum += MathF.Pow((float)(matrix[j, i] - avg), 2);
                            }
                            stds[i] = MathF.Sqrt((float)(sum / matrix.Row));
                        }
                    }
                    break;
                case 1:
                    {
                        stds = new Vector(matrix.Row);

                        means = new Vector(matrix.Row);

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            double sum = 0;
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += matrix[i, j];
                            }

                            double avg = sum / matrix.Column;

                            means[i] = avg;

                            sum = 0;

                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += MathF.Pow((float)(matrix[i, j] - avg), 2);
                            }
                            stds[i] = MathF.Sqrt((float)(sum / matrix.Column));
                        }
                    }
                    break;
                case -1:
                default:
                    {
                        stds = new Vector(1);

                        means = new Vector(1);

                        int length = (matrix.Row * matrix.Column)-1;

                        double sum = 0;

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += matrix[i, j];
                            }
                        }

                        double avg = sum / length;

                        means [0] = avg;

                        sum = 0;

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                sum += MathF.Pow((float)(matrix[i, j] - avg), 2);
                            }
                        }

                        stds[0] =  MathF.Sqrt((float)(sum /length));
                    }
                    break;
            }

            return (stds,means);
        }

        /// <summary>
        /// 协方差矩阵
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static Matrix  Cov(this Matrix matrix) 
        {
            int dimension = matrix.Column;

            int itemCount = matrix.Row;  

            int n = itemCount - 1;

            Matrix covMatrix = new Matrix(dimension, dimension);

            Matrix tempMatrix = new Matrix(itemCount, dimension);

            for (int i = 0; i < dimension; i++)
            {
                double sum = 0;

                for (int j = 0; j < itemCount; j++)
                {
                    sum += matrix[j,i];
                }

                double mean = sum / itemCount;       

                for (int j = 0; j < itemCount; j++)
                {
                    tempMatrix[j, i] = matrix[j, i] - mean;
                }
            }

            for (int i = 0; i < dimension; i++)
            {
                for (int j = 0; j < dimension; j++)
                {
                    double sum = 0;

                    for (int k = 0; k < itemCount; k++)
                    {
                        sum += tempMatrix[k,i] * tempMatrix[k,j];
                    }
                    covMatrix[i, j] = sum / n;
                }
            }


            return covMatrix;
        }

        /// <summary>
        /// 最小值最大值归一化
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="axis">-1：默认 全部归一化；0：列向量归一化；1：行向量归一化</param>
        /// <returns></returns>
        public static Matrix MinMaxScaler(this Matrix matrix,int min=0,int max=1) { 
        
            Matrix tempMatrix = Matrix.Copy(matrix);

            int s = (max - min);

            double minValue = 0;
            double maxValue = 0;

            for (int i = 0; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    double temp = matrix[i, j];

                    if (minValue > temp) minValue = temp;

                    if (maxValue < temp) maxValue = temp;
                }
            }

            for (int i = 0; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    double temp = matrix[i, j];

                    tempMatrix[i, j] = min + (temp - minValue) / (maxValue - minValue) * s;
                }
               
            }
            return tempMatrix;
        }

        /// <summary>
        /// 均值归一化
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="axis">-1：默认</param>
        /// <returns></returns>
        public static (Matrix matrix,Vector stds, Vector means) Standardized(this Matrix matrix,int axis = -1) {

            int row = matrix.Row;
            int col = matrix.Column;

            Matrix strandardMatrix = new Matrix(row, col);

            Vector stds;

            Vector means;

            switch (axis)
            {
                case 0:
                    {
                        stds = new Vector(col);

                        means = new Vector(col);

                        int n = row - 1;

                        for (int j = 0; j < col; j++)
                        {
                            double sum = 0;
                            //1.求每列均值
                            for (int i = 0; i < row; i++)
                            {
                                sum += matrix[i, j];
                            }
                            double mean = sum / row;

                            means[j] = mean;
                            //2.求每列标准差
                            sum = 0;

                            for (int i = 0; i < row; i++)
                            {
                                strandardMatrix[i, j] = matrix[i, j] - mean;

                                sum += MathF.Pow((float)strandardMatrix[i, j], 2);
                            }

                            double std = MathF.Sqrt((float)(sum / n));

                            stds[j] = std;

                            for (int i = 0; i < row; i++)
                            {
                                strandardMatrix[i, j] /= std;
                            }
                        }                        
                    }
                    break;
                case 1:
                    {

                        stds = new Vector(row);

                        means = new Vector(row);

                        int n = col - 1;

                        for (int i = 0; i < row; i++)
                        {
                            double sum = 0;
                            //1.求每列均值
                            for (int j = 0; j < col; j++)
                            {
                                sum += matrix[i, j];
                            }
                            double mean = sum / col;

                            means[i] = mean;
                            //2.求每列标准差
                            sum = 0;

                            for (int j = 0; j < col; j++)
                            {
                                strandardMatrix[i, j] = matrix[i, j] - mean;

                                sum += MathF.Pow((float)strandardMatrix[i, j], 2);
                            }

                            double std = MathF.Sqrt((float)(sum / n));

                            stds[i] = std;

                            for (int j = 0; j < col; j++)
                            {
                                strandardMatrix[i, j] /= std;
                            }
                        }
                    }

            break;
                case -1:
                default:
                    {

                        var stdResult = matrix.StandardDeviation();

                        stds = stdResult.stds;

                        means= stdResult.means;

                        double mean = means[0];

                        double std = stds[0];     

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            for (int j = 0; j < matrix.Column; j++)
                            {
                                strandardMatrix[i, j] = (matrix[i, j] - mean) / std;
                            }
                        }
                    }
                    break;
            }

           

            return (strandardMatrix, stds,means);

        }

        /// <summary>
        /// 去中心化
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="axis">-1：默认值 表示减去全体均值，0：列向量均值，1：行向量均值</param>
        /// <returns></returns>
        public static (Matrix matrix,Vector means) Centralized(this Matrix matrix,int axis = -1)
        {
            Matrix result = new Matrix(matrix.Row, matrix.Column);

            Vector means;

            switch (axis)
            {
                case 0:
                    {
                         means = matrix.Mean(axis);

                        for (int i = 0; i < matrix.Column; i++)
                        {
                            double avg = means[i];

                           

                            for (int j = 0; j < matrix.Row; j++)
                            {
                                result[j, i] = matrix[j, i] - avg;
                            }
                        }

                    }
                    break;
                case 1:
                    {
                         means = matrix.Mean(axis);                 

                        for (int i = 0; i < matrix.Row; i++)
                        {
                            double avg = means[i];

                            for (int j = 0; j < matrix.Column; j++)
                            {
                                result[i,j] = matrix[i,j] - avg;
                            }
                        }
                    }
                    break;
                case -1:
                default:
                    {
                         means = matrix.Mean(axis);
                        result = matrix - means[0];
                    }
                    break;
            }

      

           

            return (result,means);

        }


      
    }
}
