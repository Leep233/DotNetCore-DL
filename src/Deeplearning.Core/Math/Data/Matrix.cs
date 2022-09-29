using Deeplearning.Core.Exceptions;
using Deeplearning.Core.Extension;
using System;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math.Models
{
    public struct Matrix 
    {
        public const int MAX_TO_STRING_COUNT = 7;

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
        public static double Track(Matrix matrix)
        {
            int row = matrix.Row;

            int col = matrix.Column;

            int count = (int)MathF.Min(row, col);

            double temp = 0;
            for (int i = 0; i < count; i++)
            {
                temp += matrix[i, i];
            }
            return temp;
        }
        public static double Track(double[,] matrix)
        {
            int row = matrix.GetLength(0);

            int col = matrix.GetLength(1);

            int count = (int)MathF.Min(row, col);

            double temp = 0;

            for (int i = 0; i < count; i++)
            {
                temp += matrix[i, i];
            }
            return temp;
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

                            if (MathF.Abs((float)value) > 10E-15)
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
      
        public static double Det(Matrix matrix)
        {
            if (matrix.Row != matrix.Column) throw new Exception("方阵才能求行列式");

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

            return detValue;// ValueExtension.ZeroValidation();
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

                    abj[i, j] = (float)detValue;
                }
            }
            return abj;
        }

        public static Matrix Random(int row, int col)
        {
            double[,] scalers = new double[row, col];

            Random r = new Random();

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    scalers[i, j] = (float)r.NextDouble();
                }
            }
            return new Matrix(scalers);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public static Matrix DiagonalMatrix(float scalar, int row, int col)
        {
            Matrix matrix = new Matrix(row, col);

            int count = (int)MathF.Min(row, col);

            for (int i = 0; i < count; i++)
            {
                matrix[i, i] = scalar;
            }        

            return matrix;
        }

        public static Matrix DiagonalMatrix(double scalar, int size)
        {
            Matrix matrix = new Matrix(size, size);

            for (int i = 0; i < size; i++)
            {
                matrix[i, i] = scalar;
            }

            return matrix;
        }
        public static Matrix DiagonalMatrix(double[] array)
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


        public static Matrix Inv(Matrix origin)
        {
            Matrix abj = Matrix.Adjugate(origin);

            float det = (float)Matrix.Det(origin);

            if (MathF.Abs(det) <= 10E-15) {

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

            Matrix result = new Matrix(row, col);

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = m1[i, j] + scalars[j];
                }
            }        

            return result;
        }  
        public static Matrix operator +(Matrix m1, double[] scalars)
        {

            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int row = m1.Row;

            int col = m1.Column;

            Matrix result = new Matrix(row, col);

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = m1[i, j] + scalars[j];
                }
            }
 
            return result;
        }
        public static Matrix operator +(Vector vector, Matrix m1)
        {
            if (vector.Length != m1.Row)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int row = m1.Row;

            int col = m1.Column;

            Matrix result = new Matrix(row, col);

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = m1[i, j] + vector[i];
                }
            }
            return result;
        }
        public static Matrix operator +(Matrix m1, Vector vector)
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
                    result[i, j] = m1[i, j] + vector[i];
                }
            }

            return result;
        }
        public static Matrix operator +(double scalar, Matrix m1)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

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
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + scalar;
                }
            }

            return result;
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

        public static Matrix operator -(double[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalars[j] - m1[i, j];
                }
            } 

            return result;
        }
        public static Matrix operator -(Matrix m1, double[] scalars)
        {
            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = (m1[i, j] - scalars[j]);
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
                    result[i, j] = (vector[i] - m1[i, j]);
                }
            }

            return result;
        }
        public static Matrix operator -(Matrix m1, Vector vector)
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
                    result[i, j] = (m1[i, j] - vector[i]);
                }
            }

            return result;
        }
        public static Matrix operator -(double scalar, Matrix m1)
        {

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalar - m1[i, j];
                }
            }

            return result;
        }
        public static Matrix operator -(Matrix m1, double scalar)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - scalar;
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

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = (m1[i, j] * scalar);
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

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double value = m1[i, j];

                    value = value == 0 ? 0 : scalar / value;

                    result[i, j] = value;
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
