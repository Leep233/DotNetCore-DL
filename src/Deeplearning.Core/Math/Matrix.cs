﻿using Deeplearning.Core.Exceptions;
using Deeplearning.Core.Extension;
using Deeplearning.Core.Math.Common;
using System;
using System.Text;

namespace Deeplearning.Core.Math
{
    public struct Matrix 
    {
        public const int MAX_TO_STRING_COUNT = 8;

        public const string STRINGFORMAT = "F4";
        public bool IsSquare => Row == Column;

        public double[,] scalars { get; private set; }

        public double this[int row, int col]
        {
            get => scalars[row, col];
            set => scalars[row, col] = value;
        }

        public int Row { get; private set; }

        public int Column { get; private set; }

        public Matrix T => Transpose(this);       

        /// <summary>
        /// 
        /// </summary>
        /// <param name="rows">维度</param>
        /// <param name="cols">数量</param>
        public Matrix(int rows, int cols)
        {
            scalars = new double[rows, cols];

            this.Row = scalars.GetLength(0);

            this.Column = scalars.GetLength(1);
        }

        public Matrix(double[,] matrix)
        {
            this.Row = matrix.GetLength(0);

            this.Column = matrix.GetLength(1);

            scalars = matrix;
        }

        public Matrix(params Vector[] vectors)
        {
            int col = vectors.Length;

            int row = vectors[0].Length;

            scalars = new double[row, col];

            for (int r = 0; r < row; r++)
            {
                for (int c = 0; c < col; c++)
                {
                    scalars[r, c] = vectors[c][r];
                }
            }

            this.Row = scalars.GetLength(0);

            this.Column = scalars.GetLength(1);
        }

        public static double FrobeniusNorm(Matrix matrix)
        {
            return MathF.Sqrt((float)Track(matrix * matrix.T));
        }
        public static double FrobeniusNorm(double[,] matrix)
        {
            return FrobeniusNorm(new Matrix(matrix));
        }

        /// <summary>
        /// 矩阵的迹
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static double Track(Matrix matrix)
        {
            int row = matrix.Row;

            int col = matrix.Column;

            int size = (int)MathF.Min(row, col);

            double trackValue = matrix[0, 0];

            for (int i = 1; i < size; i++)
            {
                trackValue += matrix[i, i];
            }

            return trackValue;
        }
        public static double Track(double[,] matrix)
        {
            int row = matrix.GetLength(0);

            int col = matrix.GetLength(1);

            int size = (int)MathF.Min(row, col);

            double trackValue = matrix[0,0];

            for (int i = 1; i < size; i++)
            {
                trackValue += matrix[i, i];
            }
            return trackValue;
        }

        public override string ToString()
        {
            int half_count = MAX_TO_STRING_COUNT >> 1;

            string str = "...";

            StringBuilder stringBuilder = new StringBuilder();

            for (int i = 0; i < Row; i++)
            {
                if (i > half_count && Row - i > half_count)
                {
                    stringBuilder.AppendLine(str);
                }

                for (int j = 0; j < Column; j++)
                {

                    if (j > half_count && Column - j > half_count)
                    {
                        stringBuilder.AppendLine(str);
                        break;
                    }
                    else
                    {
                        stringBuilder.Append(this[i, j].ToString(STRINGFORMAT));
                        stringBuilder.Append(" ");
                    }

                }
                stringBuilder.AppendLine();
            }

            return stringBuilder.ToString();
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            bool result = true;

            if (obj is Matrix m)
            {
                if (this.Column != m.Column || this.Row != m.Row)
                {
                    result = false;
                }
                else
                {
                    for (int i = 0; i < Column; i++)
                    {
                        for (int j = 0; j < Row; j++)
                        {
                            double value = this[i, j] - m[i, j];

                            if (MathF.Abs((float)value) > MathFExtension.MIN_VALUE)
                            {
                                return false;
                            }
                        }
                    }
                }

            }
            else {

                result = false;
            }

            return result;
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
       
        public static Matrix Transpose(Matrix matrix)
        {
            int rows = matrix.Column;

            int cols = matrix.Row;

            Matrix result = new Matrix(rows, cols);

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    result[r, c] = matrix[c, r];
                }
            }
            return result;
        }


        public static Matrix HadamardProduct(Matrix matrix01, Matrix matrix02)
        {
            if (matrix01.Row != matrix02.Row || matrix01.Column != matrix02.Column)
                throw new ArgumentException("矩阵大小不一致，无法相加");

            int row = matrix01.Row;
            int col = matrix01.Column;
            Matrix result = new Matrix(row, col);     

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = matrix01[i, j] * matrix02[i, j];
                }
            }
            return result;
        }
      
            /// <summary>
            /// 矩阵行列式
            /// </summary>
            /// <param name="matrix"></param>
            /// <returns></returns>
            /// <exception cref="Exception"></exception>
        public static double Det(Matrix matrix)
        {
            if (!matrix.IsSquare) throw new Exception("方阵才能求行列式");

            double detValue = 0;

            int size = matrix.Row;

            switch (size)
            {
                case 0:

                    break;
                case 1:
                    {
                        detValue = matrix[0, 0];
                    }
                    break;
                case 2:
                    {
                        detValue = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
                    }
                    break;
                default:
                    {
                        for (int i = 0; i < size; i++)
                        {
                            double scalarValue = matrix[0, i];

                            Matrix cofactor = matrix.AlgebraicCofactor(0, i);
                            double value = ((i + 2) % 2 == 0 ? 1 : -1) * scalarValue * Det(cofactor);
                            detValue += value;
                        }
                    }
                    break;
            }

            return detValue;
        }

        /// <summary>
        /// 代数余子式
        /// </summary>
        /// <param name="rowIndex">对应元素的行下标</param>
        /// <param name="colIndex">对应元素的列下标</param>
        /// <returns></returns>
        public Matrix AlgebraicCofactor(int rowIndex, int colIndex)
        {
            int rowCount = rowIndex < 0 ? Row : Row - 1;
            int colCount = colIndex < 0 ? Column : Column - 1;
            Matrix matrix = new Matrix(rowCount, colCount);

            int r_i = 0;

            for (int i = 0; i < Row; i++)
            {
                if (i == colIndex) continue;

                int r_j = 0;

                for (int j = 0; j < Column; j++)
                {
                    if (j == rowIndex) continue;

                    matrix[r_i, r_j] = scalars[i, j];

                    r_j++;
                }
                r_i++;
            }
            return matrix;
        }

        /// <summary>
        /// 伴随矩阵
        /// </summary>
        /// <param name="origin"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Matrix Adjugate(Matrix origin)
        {

            if (origin.Row != origin.Column) throw new ArgumentException("非方阵无法求伴随矩阵");

            int size = origin.Row;

            Matrix abj = new Matrix(size, size);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    Matrix temp = origin.AlgebraicCofactor(i, j);

                    double detValue = (((i + j + 2) % 2 == 0 ? 1 : -1)) * Det(temp);

                    abj[i, j] = detValue;
                }
            }
            return abj;
        }

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
       /// 获取某一列向量
       /// </summary>
       /// <param name="index">向量所在列</param>
       /// <returns></returns>
       /// <exception cref="ArgumentOutOfRangeException"></exception>
        public Vector GetVector(int index)
        {
            int row = this.Row;

            int col = this.Column;    

            if (col <= index) throw new ArgumentOutOfRangeException("越界");

            Vector vector = new Vector(row);

            for (int i = 0; i < row; i++)
            {
                vector[i] = this[i, index];
            }
            return vector;
        }

        /// <summary>
        /// 获取矩阵所有列向量
        /// </summary>
        /// <param name="vectorType"></param>
        /// <returns></returns>
        public Vector[] ToVectors()
        {
            int row = this.Row;

            int col = this.Column;
       
            Vector[] vs = new Vector[col];

            for (int c = 0; c < col; c++)
            {
                vs[c] = new Vector(row);

                for (int r = 0; r < row; r++)
                {
                    vs[c][r] = this[r, c];
                }
            }
            return vs;
        }


        /// <summary>
        /// 求均值向量
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static Vector Average(Matrix source)
        {

            int vectorCount = source.Column;

            int vectorLength = source.Row;
           
            Vector avgs = new Vector(vectorLength);

            for (int r = 0; r < vectorLength; r++)
            {
                double sum = 0;
                for (int c = 0; c < vectorCount; c++)
                {
                    sum += source[r, c];
                }
                avgs[r] = sum / vectorCount;
            }

            return avgs;
        }

        /// <summary>
        /// 矩阵归一化
        /// </summary>
        /// <param name="source"></param>
        /// <param name="mode">0，将所有列向量归一，其他将行向量归一</param>
        /// <returns></returns>
        public static Matrix Normalized(Matrix source)
        {
            int vectorCount = source.Column;

            int vectorLength = source.Row;

            Matrix matrix = new Matrix(vectorLength, vectorCount);

            //求出了每一行的均值
            Vector avgs = Average(source);
    
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
                    matrix[r, c] = (source[r, c] - avgs[r]) / sdevs[r];
                }
            }

            return matrix;
        }

        /// <summary>
        /// 获取主对角所有元素
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static double[] DiagonalElements(Matrix source)
        {

            int r = (int)MathF.Min(source.Row, source.Column);

            double[] vector = new double[r];

            for (int i = 0; i < r; i++)
            {
                vector[i] = source[i, i];
            }
            return vector;
        }


        /// <summary>
        /// 协方差
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static Matrix Cov(Matrix source)
        {

            float n = source.Column - 1;//;//

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
                    result[i, j] = temp/ n;
                }
            }
            return result;
        }

        /// <summary>
        /// 均值归一化，
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static (Matrix matrix, Vector avgs,Vector stds) MeanNormalization(Matrix matrix)
        {
            int row = matrix.Row;

            int col = matrix.Column;

            Matrix normMatrix = new Matrix(row, col);

            Vector avgs = Average(matrix);

            Vector vars = new Vector(row);

            int n = row-1;

            for (int r = 0; r < row; r++)
            {
                double sum = 0;
                double avg = avgs[r];
                for (int c = 0; c < col; c++)
                {
                    sum += MathF.Pow((float)(matrix[r,c] - avg),2);
                }
                vars[r] = MathF.Sqrt((float)(sum / n));// sum / n;// MathF.Sqrt((float)(sum / n));
            }

            for (int r = 0; r < row; r++)
            {
                double avg = avgs[r];
                double var = vars[r];
                for (int c = 0; c < col; c++)
                {
                    normMatrix[r, c] = (matrix[r, c] - avg)/ var;
                }
            }

            return (normMatrix, avgs, vars);

        }
     
        /// <summary>
        /// 方差
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static double[] Var(Matrix source)
        {
            int row = source.Row;

            int col = source.Column;

            Vector avgs = Average(source);       

            double[] vars = new double[row];

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


            return vars;
        }
       
        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static double[] StandardDeviation(Matrix source)
        {

            int row = source.Row;

            int col = source.Column;

            Vector avgs = Average(source);

            double[] sdevs = null;

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
                sdevs[r] = MathF.Sqrt((float)(sum / n));
            }

            return sdevs;
        }

        public static Matrix Replace(Matrix source, Vector vector, int colIndex)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                source[i, colIndex] = vector[i];
            }

            return source;
        }


        /// <summary>
        /// 中心化（零均值化）：是指变量减去它的均值。其实就是一个平移的过程，平移后所有数据的中心是（0，0）
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static (Matrix matrix, Vector avgs) Centralized(Matrix source)
        {
            int vectorCount = source.Column;

            int vectorLength = source.Row;

            Matrix matrix = new Matrix(vectorLength, vectorCount);

            Vector avgs = Average(source);

            //for (int r = 0; r < vectorLength; r++)
            //{
            //    double sum = 0;
            //    for (int c = 0; c < vectorCount; c++)
            //    {
            //        sum += source[r, c];
            //    }
            //    avgs[r] = sum / vectorCount;
            //}

            for (int r = 0; r < vectorLength; r++)
            {
                for (int c = 0; c < vectorCount; c++)
                {
                    matrix[r, c] = source[r, c] - avgs[r];
                }
            }

            return (matrix, avgs);

        }

        /// <summary>
        /// 判断是否是对称矩阵
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static bool Symmetry(Matrix source)
        {
            if (source.Row != source.Column) return false;

            int row = source.Row;

            for (int r = 0; r < row; r++)
            {
                for (int c = 0; c < row; c++)
                {
                    if (c == r) continue;

                    if (source[r, c] != source[c, r]) return false;
                }
            }
            return true;
        }

        public static Matrix Replace(Matrix source, Matrix matrix, int offsetR, int offsetC)
        {
            if (offsetR >= source.Row || offsetC >= source.Column) throw new IndexOutOfRangeException();

            int rowCount = matrix.Row;// - rCount;
            int colCount = matrix.Column;// - cCount;

            for (int r = 0; r < rowCount; r++)
            {
                for (int c = 0; c < colCount; c++)
                {
                    source[r + offsetR, c + offsetC] = matrix[r, c];
                }
            }
            return source;

        }
       
        /// <summary>
        /// 裁剪矩阵
        /// </summary>
        /// <param name="source"></param>
        /// <param name="startRow"></param>
        /// <param name="startColumn"></param>
        /// <param name="endRow"></param>
        /// <param name="endColumn"></param>
        /// <returns></returns>
        public static Matrix Clip(Matrix source, int startRow, int startColumn, int endRow, int endColumn)
        {
            int sR = (int)MathF.Max(startRow, 0); //startRow < 0 ? 0 : startRow;
            int sC = (int)MathF.Max(startColumn, 0); //startColumn < 0 ? 0 : startColumn;

            int eR = (int)MathF.Min(endRow, source.Row); //endRow <= source.Row ? endRow : source.Row;
            int eC = (int)MathF.Min(endColumn, source.Column); //endColumn <= source.Column ? endColumn : source.Column;

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

        /// <summary>
        /// 矩阵逆
        /// </summary>
        /// <param name="origin"></param>
        /// <returns></returns>
        /// <exception cref="SingularMatrixException"></exception>
        public static Matrix Inv(Matrix origin)
        {
            Matrix abj = Adjugate(origin);

            float det = (float)Det(origin);

            if (MathF.Abs(det) <= MathFExtension.MIN_VALUE) 
            {
                throw new SingularMatrixException();
            }

            return abj / det;
        }


        /// <summary>
        /// 初等行变化
        /// </summary>
        /// <param name="matrix"></param>
        public static Matrix ElementaryTransformation(Matrix coefficientMatrix)
        {
            Matrix matrix = coefficientMatrix;

            int row = matrix.Row;

            int col = matrix.Column;

            //有的时候 矩阵只需要排序就可以变成阶梯型 
            //行排序
            int lastIndex = row - 1;

            for (int i = 0; i < lastIndex; i++)
            {
                for (int c = i; c < col; c++)
                {
                    if (matrix[i, c] == 0)
                    {
                        for (int j = 0; j < col; j++)
                        {
                            double t = matrix[lastIndex, j];
                            matrix[lastIndex, j] = matrix[i, j];
                            matrix[i, j] = t;
                        }
                    }
                }
            }
            //行排序 结束

            double mid = 0;

            for (int i = 0; i < lastIndex; i++)
            {
                mid = matrix[i, i];

                if (mid != 0 && mid != 1)
                {
                    for (int j = i; j < col; j++)
                    {
                        double temp = matrix[i, j];

                        temp = mid == 0 ? temp : temp / mid;

                        matrix[i, j] = temp;
                    }
                }

                for (int j = i + 1; j < row; j++)
                {
                    mid = matrix[j, i];

                    for (int q = i; q < col; q++)
                    {
                        double value = mid * matrix[i, q];

                        matrix[j, q] -= value;
                    }
                }
            }
            return matrix;
        }


        #region 运算符重载

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            int rows = (int)MathF.Min(m1.Row, m2.Row);

            int cols = (int)MathF.Min(m1.Column, m2.Column);

            int MaxRows = (int)MathF.Max(m1.Row, m2.Row);

            int MaxCols = (int)MathF.Max(m1.Column, m2.Column);

            Matrix result = new Matrix(MaxRows, MaxCols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + m2[i, j];
                }
            }

            return result;
        }

        public static Matrix operator +(double[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int row = m1.Row;

            int col = m1.Column;

            Matrix result = m1;

            for (int j = 0; j < col; j++)
            {
                double temp = scalars[j];

                if (MathF.Abs((float)temp) <= MathFExtension.MIN_VALUE) continue;

                for (int i = 0; i < row; i++)
                {
                    result[i, j] += temp;
                }
            }
            return result;
        }
        public static Matrix operator +(Matrix m1, double[] scalars)
        {
            return scalars+m1;
        }
        public static Matrix operator +(Vector vector, Matrix m1)
        {
            if (vector.Length != m1.Row)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int row = m1.Row;

            int col = m1.Column;

            Matrix result = m1;

            for (int i = 0; i < row; i++)
            {
                double temp = vector[i];

                if (MathF.Abs((float)temp) <= MathFExtension.MIN_VALUE) continue;

                for (int j = 0; j < col; j++)
                {
                    result[i, j] += temp;
                }
            }
           
            return result;
        }
        public static Matrix operator +(Matrix m1, Vector vector)
        {
            return vector + m1;
        }
        public static Matrix operator +(double scalar, Matrix m1)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = m1;       

            if(MathF.Abs((float)scalar)<=MathFExtension.MIN_VALUE) return result;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + scalar;
                }

            }
            return result;
        }
        public static Matrix operator +(Matrix m1, double scalar)
        {
            return scalar+m1;
        }
        public static Matrix operator -(Matrix m1, Matrix m2)
        {

            int rows = (int)MathF.Min(m1.Row, m2.Row);

            int cols = (int)MathF.Min(m1.Column, m2.Column);

            int MaxRows = (int)MathF.Max(m1.Row, m2.Row);

            int MaxCols = (int)MathF.Max(m1.Column, m2.Column);

            Matrix result = new Matrix(MaxRows, MaxCols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - m2[i, j];
                }
            }

            return result;
        }

        public static Matrix operator -(double[] scalars, Matrix matrix)
        {
            if (scalars.Length != matrix.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = matrix.Row;

            int cols = matrix.Column;

            Matrix result = matrix;

            for (int j = 0; j < cols; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    result[i, j] = scalars[j] - matrix[i, j];
                }
            }
            return result;
        }
        public static Matrix operator -(Matrix matrix, double[] scalars)
        {
            if (scalars.Length != matrix.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = matrix.Row;

            int cols = matrix.Column;

            Matrix result = matrix;
            for (int j = 0; j < cols; j++)
            {
                double temp = scalars[j];

                if (MathF.Abs((float)temp) <= MathFExtension.MIN_VALUE) continue;

                for (int i = 0; i < rows; i++)
                {
                    result[i, j] -= temp;
                }
            }
            return result;
        }

        public static Matrix operator -(Vector vector, Matrix m1)
        {

            if (vector.Length != m1.Row)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = vector[i] - m1[i, j];
                }
            }

            return result;
        }
        public static Matrix operator -(Matrix matrix, Vector vector)
        {
            if (vector.Length != matrix.Row)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = matrix.Row;

            int cols = matrix.Column;

            Matrix result = matrix;

            for (int i = 0; i < rows; i++)
            {
                double temp = vector[i];

                if (MathF.Abs((float)temp) <= MathFExtension.MIN_VALUE) continue;

                for (int j = 0; j < cols; j++)
                {
                    result[i, j] -= temp;
                }
            }

            return result;
        }
        public static Matrix operator -(double scalar, Matrix matrix)
        {

            int rows = matrix.Row;

            int cols = matrix.Column;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalar - matrix[i, j];
                }
            }

            return result;
        }
        public static Matrix operator -(Matrix m1, double scalar)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = m1;

            if (MathF.Abs((float)scalar) <= MathFExtension.MIN_VALUE) return result;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] -= scalar;
                }
            }
            return result;
        }
        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            if (m1.Column != m2.Row)
                throw new ArgumentException("矩阵大小不一致，无法相乘");

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
                    result[i, j] = temp;
                }
            }
              
            return result;
        }


        public static double[] operator *(double[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            double[] result = new double[cols];

            for (int i = 0; i < rows; i++)
            {
                double temp = 0;
                for (int j = 0; j < rows; j++)
                {
                    temp += m1[j, i] * scalars[j];
                }
                result[i] = temp;
            }

            return result;
        }

        public static Vector operator *(Matrix m1, Vector vector)
        {

            if (vector.Length != m1.Column)
                throw new ArgumentException("向量维度与矩阵行数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Vector result = new Vector(rows);

            for (int i = 0; i < rows; i++)
            {
                double temp = 0;

                for (int j = 0; j < cols; j++)
                {
                    temp += m1[i, j] * vector[j];
                }
                result[i] = temp;
            }
            return result;
        }
        public static Matrix operator *(double scalar, Matrix m1)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            if (MathF.Abs((float)scalar) <= MathFExtension.MIN_VALUE) return result;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalar * m1[i, j];
                }
            }

            return result;
        }

        public static Matrix operator *(Matrix m1, double scalar)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            if (MathF.Abs((float)scalar) <= MathFExtension.MIN_VALUE) return result;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] * scalar;
                }
            }

            return result;
        }
    
        public static Matrix operator /(Matrix m1, double scalar)
        {
            if (scalar == 0)
                throw new ArgumentException("被除数不能为零");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] / scalar;
                }
            }

            return result;
        }
        public static Matrix operator /(double scalar, Matrix m1)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            if (MathF.Abs((float)scalar) <= MathFExtension.MIN_VALUE) return result;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalar / m1[i, j];
                }
            }

            return result;
        }

        public static bool operator ==(Matrix m1, Matrix m2)
        {
            return m1.Equals(m2);
        }
        public static bool operator !=(Matrix m1, Matrix m2)
        {
            return !m1.Equals(m2);
        }
        #endregion

    }
}
