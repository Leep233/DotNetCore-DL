using System;
using System.Text;

namespace Deeplearning.Core.Math.Models
{
    public class Matrix : ICloneable
    {
        public const int MAX_TO_STRING_COUNT = 4;

        public const float MinValue = 0.0001f;

        private float[,] scalar;

        public float this[int row, int col]
        {
            get => scalar[col, row];
            set => scalar[col, row] = value;
        }

        public int Rows { get; private set; } = 0;

        public int Columns { get; private set; } = 0;

        public Matrix T => Transpose(this);
        public float det => Det(this);
      
        /// <summary>
        /// 
        /// </summary>
        /// <param name="rows">维度</param>
        /// <param name="cols">数量</param>
        public Matrix(int rows, int cols)
        {
            this.Rows = rows;
            this.Columns = cols;
            scalar = new float[Rows, Columns];
        }

        public Matrix(float[,] matrix)
        {
            this.Rows = matrix.GetLength(0);
            this.Columns = matrix.GetLength(1);
            scalar = matrix;
        }
      
        public Matrix(Vector[] vectors)
        {
            this.Rows = vectors.Length;
            this.Columns = vectors[0].Length;

            scalar = new float[Rows, Columns];

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    scalar[i, j] = vectors[i][j];
                }
            }

        }

        private Matrix()
        {

        }

        public object Clone()
        {
            return Copy(this);
        }

        public float FrobeniusNorm()
        {
            return MathF.Sqrt(Track(this * T));
        }

        public float Track() { 
        
               return Track(this);
        
        }

        public override string ToString()
        {
            int half_count = MAX_TO_STRING_COUNT >> 1;

            string str = "...";

            StringBuilder stringBuilder = new StringBuilder();

            for (int i = 0; i < Rows; i++)
            {
                if (i > half_count && Rows - i > half_count)
                {
                    stringBuilder.AppendLine(str);
                }

                for (int j = 0; j < Columns; j++)
                {

                    if (j > half_count && Columns - j > half_count)
                    {
                        stringBuilder.AppendLine(str);
                        break;
                    }
                    else
                    {
                        stringBuilder.Append(this[i, j].ToString("F4"));
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

            Matrix m = obj as Matrix;

            if (this.Columns != m.Columns || this.Rows != m.Rows)
            {
                result = false;
            }
            else
            {
                for (int i = 0; i < Columns; i++)
                {
                    for (int j = 0; j < Rows; j++)
                    {
                        if (this[i, j] != m[i, j]) {
                            result = false;
                            break;
                        }
                    }
                }
            }          
            return result;
        }

        /// <summary>
        /// 是否正交
        /// </summary>
        /// <returns></returns>
        public  bool IsOrthogonal() 
        {
            if (!IsSquare())
                return false;

            Matrix unitMatrix = UnitMatrix(Rows);

            Matrix matrix = this.T * this;

            return unitMatrix.Equals(matrix);
        }
        public bool IsSquare() {

            return Rows == Columns;

        }
        public static Matrix DiagonalMatrix(float scalar,int size)
        {
    
            Matrix matrix = new Matrix(size, size);

            for (int i = 0; i < matrix.Rows; i++)
            {
                matrix[i, i] = scalar;
            }

            return matrix;
        }
        public static Matrix DiagonalMatrix(float[] array)
        {
            int size = array.Length;

            Matrix matrix = new Matrix(size, size);

            for (int i = 0; i < matrix.Rows; i++)
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

            Matrix result = new Matrix(matrix.Columns, matrix.Rows);

            for (int i = 0; i < result.Columns; i++)
            {
                for (int j = 0; j < result.Rows; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }
            return result;
        }

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            if (m1.Rows != m2.Rows || m1.Columns != m2.Columns)
                throw new ArgumentException("矩阵大小不一致，无法相加");

            int rows = m1.Rows;

            int cols = m1.Columns;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + m2[i, j];
                }
            }
            return result;
        }
        public static Matrix operator +(float[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Rows)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + scalars[j];
                }
            }
            return result;
        }
        public static Matrix operator +(Matrix m1, float[] scalars)
        {

            if (scalars.Length != m1.Rows)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + scalars[j];
                }
            }
            return result;
        }
        public static Matrix operator +(Vector vector, Matrix m1)
        {

            if (vector.Length != m1.Rows)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

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
        public static Matrix operator +(Matrix m1, Vector vector)
        {

            if (vector.Length != m1.Rows)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

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

        public static Matrix operator +(float scalar, Matrix m1)
        {

            int rows = m1.Rows;

            int cols = m1.Columns;

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
        public static Matrix operator +(Matrix m1, float scalar)
        {
            int rows = m1.Rows;

            int cols = m1.Columns;

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
            if (m1.Rows != m2.Rows || m1.Columns != m2.Columns)
                throw new ArgumentException("矩阵大小不一致，无法相加");

            int rows = m1.Rows;

            int cols = m1.Columns;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - m2[i, j];
                }
            }
            return result;
        }
        public static Matrix operator -(float[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Rows)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

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
        public static Matrix operator -(Matrix m1, float[] scalars)
        {

            if (scalars.Length != m1.Rows)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - scalars[j];
                }
            }
            return result;
        }

        public static Matrix operator -(Vector vector, Matrix m1)
        {

            if (vector.Length != m1.Rows)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

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
        public static Matrix operator -(Matrix m1, Vector vector)
        {

            if (vector.Length != m1.Rows)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - vector[i];
                }
            }
            return result;
        }
        public static Matrix operator -(float scalar, Matrix m1)
        {

            int rows = m1.Rows;

            int cols = m1.Columns;

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
        public static Matrix operator -(Matrix m1, float scalar)
        {
            int rows = m1.Rows;

            int cols = m1.Columns;

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
            if (m1.Columns != m2.Rows)
                throw new ArgumentException("矩阵大小不一致，无法相加");

            int rows = m1.Rows;

            int same = m1.Columns;

            int cols = m2.Columns;

            float temp = 0;

            Matrix result = new Matrix(m1.Rows, m2.Columns);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    temp = 0;

                    for (int k = 0; k < same; k++)
                    {
                        temp += m1[i, k] * m2[k, j];
                    }
                    result[i, j] = temp;                   
                }
            }
            return result;
        }
        public static Vector operator *(float[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Rows)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

             Vector  result = new Vector(cols);

            for (int i = 0; i < cols; i++)
            {
                float temp = 0;
                for (int j = 0; j < rows; j++)
                {
                    temp += m1[j, i] * scalars[j];
                }
                result[i] = temp;
            }

            return result;
        }
        public static float[] operator *(Matrix m1, Vector vector)
        {

            if (vector.Length != m1.Columns)
                throw new ArgumentException("向量维度与矩阵行数不一致，无法操作");

            int rows = m1.Rows;

            int cols = m1.Columns;

            float[] result = new float[rows];

            for (int i = 0; i < rows; i++)
            {
                float temp = 0;
                for (int j = 0; j < cols; j++)
                {
                    temp+= m1[i, j] * vector[j];
                }
                result[i] = temp;
            }
            return result;
        }
        public static Matrix operator *(float scalar, Matrix m1)
        {
            int rows = m1.Rows;

            int cols = m1.Columns;

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
        public static Matrix operator *(Matrix m1, float scalar)
        {
            int rows = m1.Rows;

            int cols = m1.Columns;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] * scalar;
                }
            }
            return result;
        }
        public static Matrix operator /(Matrix m1, float scalar)
        {
            int rows = m1.Rows;

            int cols = m1.Columns;

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

        public static Matrix operator /(float scalar,Matrix m1)
        {
            int rows = m1.Rows;

            int cols = m1.Columns;

            Matrix result = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalar/ m1[i, j] ;
                }
            }
            return result;
        }

        public static bool operator ==(Matrix m1, Matrix m2) {
            return m1.Equals(m2);        
        }

        public static bool operator !=(Matrix m1, Matrix m2)
        {
            return !m1.Equals(m2);
        }

        public static Matrix HadamardProduct(Matrix m1, Matrix m2)
        {

            if (m1.Rows != m2.Rows || m1.Columns != m2.Columns)
                throw new ArgumentException("矩阵大小不一致，无法相加");

            int row = m1.Rows;
            int col = m1.Columns;
            Matrix result = new Matrix(row, col);

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = m1[i, j] * m2[i, j];
                }
            }
            return result;
        }

        public static float FrobeniusNorm(Matrix matrix)
        {
            return MathF.Sqrt(Track(matrix * matrix.T));
        }

        public static float FrobeniusNorm(float[,] matrix)
        {
            return FrobeniusNorm(new Matrix(matrix));
        }
        public static float Track(Matrix matrix)
        {
            int row = matrix.Rows;

            int col = matrix.Columns;

            int count = row > col ? col : row;

            float temp = 0;

            for (int i = 0; i < count; i++)
            {
                temp += matrix[i, i];
            }
            return temp;
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

        /// <summary>
        /// 获取余子式
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <param name="source"></param>
        /// <returns></returns>
        private static Matrix GetMinor(int row, int col, Matrix source)
        {

            int rows = source.Rows;

            int cols = source.Columns;

            int rowsLength = rows - 1;

            int colsLength = cols - 1;

            int m_i_index = 0;

            int m_j_index = 0;

            //余子式
            Matrix minor = new Matrix(rowsLength, colsLength);

            for (int m = 0; m < rows; m++)
            {
                if (m == row) continue;

                m_j_index = 0;

                for (int n = 0; n < cols; n++)
                {
                    if (n == col) continue;

                    minor[m_i_index, m_j_index] = source[m, n];

                    m_j_index++;
                }

                m_i_index++;
            }

            return minor;
        }

        public static Matrix Copy(Matrix source)
        {
         

            Matrix matrix = new Matrix(source.Rows, source.Columns);

            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = source[i, j];
                }
            }
            return matrix;
        }

        public static float Det(Matrix matrix)
        {
            float det = 0;

            int rows = matrix.Rows;

            int cols = matrix.Columns;

            if (rows == 2 && cols == 2)
            {
                det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
            }
            else
            {
                int size = rows - 1;

                Matrix minor = new Matrix(size, size);

                for (int i = 0; i < cols; i++)
                {
                    int c = 0;

                    minor = GetMinor(c, i, matrix);

                    int n = (i + 2) % 2 == 0 ? 1 : -1;

                    det += n * matrix[c, i] * Det(minor);
                }
            }
            return det;
        }

        public  Matrix Adjoint() 
        {
            float det = 1;

            int rows = Rows;

            int cols = Columns;

            Matrix matrix = new Matrix(rows, cols);


            if (rows == 2 && cols == 2)
            {
                det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
            }
            else
            {
                int size = rows - 1;

                Matrix minor = new Matrix(size, size);

                for (int i = 0; i < cols; i++)
                {
                    int c = 0;

                    minor = GetMinor(c, i, matrix).T;

                    int n = (i + 2) % 2 == 0 ? 1 : -1;

                    det += n * matrix[c, i] * Det(minor);

                    matrix = minor / det;
                }
            }
            return matrix;
        }

        public static Matrix Inv(Matrix matrix) {

            // for

            return null;

        }

    }
}
