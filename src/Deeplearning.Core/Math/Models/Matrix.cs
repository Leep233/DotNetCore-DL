using System;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math.Models
{
    public class Matrix : ICloneable
    {
        public const int MAX_TO_STRING_COUNT = 7;

        public const string STRINGFORMAT = "F12";
        public bool IsSquare => Row == Column;

        public float[,] scalars { get; private set; }

        public float this[int row, int col]
        {
            get => scalars[row, col];
            set => scalars[row, col] = value;
        }

        public int Row { get; private set; } = 0;

        public int Column { get; private set; } = 0;

        public Matrix T => Transpose(this);
        public float det => Det(this);
        public Matrix abj => Adjugate(this);
        public Matrix inverse => Inv(this);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="rows">维度</param>
        /// <param name="cols">数量</param>
        public Matrix(int rows, int cols)
        {
            scalars = new float[rows, cols];

            this.Row = scalars.GetLength(0);

            this.Column = scalars.GetLength(1);
        }

        public Matrix(float[,] matrix)
        {
            this.Row = matrix.GetLength(0);

            this.Column = matrix.GetLength(1);

            scalars = matrix;
        }

        public Matrix(params Vector[] vectors)
        {
            int col = vectors.Length;

            int row = vectors[0].Length;

            scalars = new float[row, col];

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

        private Matrix()
        {

        }


        public double FrobeniusNorm()
        {
            return MathF.Sqrt((float)Track(this * T));
        }

        public double Track()
        {
            return Track(this);
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

            Matrix m = obj as Matrix;

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

                        if (Validator.ZeroValidation(value) != 0)
                        {
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
        public bool IsOrthogonal()
        {
            if (!IsSquare)
                return false;

            Matrix unitMatrix = UnitMatrix(Row);

            Matrix matrix = this.T * this;

            return unitMatrix.Equals(matrix);
        }

        public static Matrix UnitMatrix(int size)
        {
            Matrix matrix = new Matrix(size, size);

            Parallel.For(0, size, i => {
                matrix[i, i] = 1;
            });

            //for (int i = 0; i < size; i++)
            //{
            //    matrix[i, i] = 1;
            //}
            return matrix;
        }
        public static Matrix Transpose(Matrix matrix)
        {
            int rows = matrix.Column;

            int cols = matrix.Row;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, r =>
            {
                for (int c = 0; c < cols; c++)
                {
                    result[r, c] = matrix[c, r];
                }
            });

            //for (int r = 0; r < rows; r++)
            //{
            //    for (int c = 0; c < cols; c++)
            //    {
            //        result[r, c] = matrix[c, r];
            //    }
            //}
            return result;
        }
        public static Matrix HadamardProduct(Matrix m1, Matrix m2)
        {

            if (m1.Row != m2.Row || m1.Column != m2.Column)
                throw new ArgumentException("矩阵大小不一致，无法相加");

            int row = m1.Row;
            int col = m1.Column;
            Matrix result = new Matrix(row, col);

            Parallel.For(0, row, i =>
            {
                for (int j = 0; j < col; j++)
                {
                    result[i, j] = m1[i, j] * m2[i, j];
                }
            });

            //for (int i = 0; i < row; i++)
            //{
            //    for (int j = 0; j < col; j++)
            //    {
            //        result[i, j] = m1[i, j] * m2[i, j];
            //    }
            //}
            return result;
        }
        public static double FrobeniusNorm(Matrix matrix)
        {
            return MathF.Sqrt((float)Track(matrix * matrix.T));
        }
        public static double FrobeniusNorm(float[,] matrix)
        {
            return FrobeniusNorm(new Matrix(matrix));
        }
        public static double Track(Matrix matrix)
        {
            int row = matrix.Row;

            int col = matrix.Column;

            int count = (int)MathF.Min(row, col);

            double temp = 0;

            Parallel.For(0, count, i => { temp += matrix[i, i]; });

            //for (int i = 0; i < count; i++)
            //{
            //    temp += matrix[i, i];
            //}
            return temp;
        }
        public static double Track(double[,] matrix)
        {
            int row = matrix.GetLength(0);

            int col = matrix.GetLength(1);

            int count = (int)MathF.Min(row, col);

            double temp = 0;

            Parallel.For(0, count, i => { temp += matrix[i, i]; });
            //for (int i = 0; i < count; i++)
            //{
            //    temp += matrix[i, i];
            //}
            return temp;
        }
        public object Clone()
        {
            return Copy(this);
        }
        public static Matrix Copy(Matrix source)
        {
            Matrix matrix = new Matrix(source.Row, source.Column);

            for (int i = 0; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    matrix[i, j] = source[i, j];
                }
            }
            return matrix;
        }

        public static float Det(Matrix matrix)
        {
            if (matrix.Row != matrix.Column) throw new Exception("方阵才能求行列式");

            float detValue = 0;

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
                            float scalarValue = matrix[0, i];

                            Matrix cofactor = matrix.AlgebraicCofactor(0, i);

                            detValue += ((i + 2) % 2 == 0 ? 1 : -1) * scalarValue * Det(cofactor);
                        }
                    }
                    break;
            }

            return Validator.ZeroValidation(detValue);
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

        public static Matrix Adjugate(Matrix origin)
        {

            if (!origin.IsSquare) return null;

            int size = origin.Row;

            Matrix abj = new Matrix(size, size);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    Matrix temp = origin.AlgebraicCofactor(i, j);

                    float detValue = (((i + j + 2) % 2 == 0 ? 1 : -1)) * temp.det;

                    abj[i, j] = detValue;
                }
            }
            return abj;
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

            //int count = row < col ? row : col;

            int count = (int)MathF.Min(row, col);

            Parallel.For(0, count, i => { matrix[i, i] = scalar; });

            //for (int i = 0; i < count; i++)
            //{
            //    matrix[i, i] = scalar;
            //}

            return matrix;
        }

        public static Matrix DiagonalMatrix(float scalar, int size)
        {

            Matrix matrix = new Matrix(size, size);

            Parallel.For(0, size, i => { matrix[i, i] = scalar; });

            //for (int i = 0; i < matrix.Row; i++)
            //{
            //    matrix[i, i] = scalar;
            //}

            return matrix;
        }
        public static Matrix DiagonalMatrix(float[] array)
        {
            int size = array.Length;

            Matrix matrix = new Matrix(size, size);

            Parallel.For(0, size, i => { matrix[i, i] = array[i]; });

            //for (int i = 0; i < matrix.Row; i++)
            //{
            //    matrix[i, i] = array[i];
            //}

            return matrix;
        }
        public static Matrix DiagonalMatrix(Vector vector)
        {
            int size = vector.Length;

            Matrix matrix = new Matrix(size, size);


            Parallel.For(0, size, i => { matrix[i, i] = vector[i]; });

            //for (int i = 0; i < size; i++)
            //{
            //    matrix[i, i] = vector[i];
            //}

            return matrix;
        }


        public static Matrix Inv(Matrix origin)
        {
            return origin.abj / origin.det;
        }

        #region 运算符重载

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            int rows = (int)MathF.Min(m1.Row, m2.Row);

            int cols = (int)MathF.Min(m1.Column, m2.Column);

            int MaxRows = (int)MathF.Max(m1.Row, m2.Row);

            int MaxCols = (int)MathF.Max(m1.Column, m2.Column);

            Matrix result = new Matrix(MaxRows, MaxCols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + m2[i, j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] + m2[i, j];
            //    }
            //}
            return result;
        }
        public static Matrix operator +(float[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + scalars[j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] + scalars[j];
            //    }
            //}
            return result;
        }
        public static Matrix operator +(Matrix m1, float[] scalars)
        {

            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + scalars[j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] + scalars[j];
            //    }
            //}
            return result;
        }
        public static Matrix operator +(Vector vector, Matrix m1)
        {

            if (vector.Length != m1.Row)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + vector[i];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] + vector[i];
            //    }
            //}
            return result;
        }
        public static Matrix operator +(Matrix m1, Vector vector)
        {

            if (vector.Length != m1.Row)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);



            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + vector[i];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] + vector[i];
            //    }
            //}
            return result;
        }
        public static Matrix operator +(float scalar, Matrix m1)
        {

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + scalar;
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] + scalar;
            //    }
            //}
            return result;
        }
        public static Matrix operator +(Matrix m1, float scalar)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);


            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] + scalar;
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] + scalar;
            //    }
            //}
            return result;
        }
        public static Matrix operator -(Matrix m1, Matrix m2)
        {

            int rows = (int)MathF.Min(m1.Row, m2.Row);

            int cols = (int)MathF.Min(m1.Column, m2.Column);

            int MaxRows = (int)MathF.Max(m1.Row, m2.Row);

            int MaxCols = (int)MathF.Max(m1.Column, m2.Column);

            Matrix result = new Matrix(MaxRows, MaxCols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - m2[i, j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] - m2[i, j];
            //    }
            //}
            return result;
        }
        public static Matrix operator -(float[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);


            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalars[j] - m1[i, j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = scalars[j] - m1[i, j];
            //    }
            //}
            return result;
        }
        public static Matrix operator -(Matrix m1, float[] scalars)
        {
            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - scalars[j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {      
            //        result[i, j] = m1[i, j] - scalars[j];
            //    }
            //}
            return result;
        }
        public static Matrix operator -(Vector vector, Matrix m1)
        {

            if (vector.Length != m1.Row)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = vector[i] - m1[i, j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {                   
            //        result[i, j] = vector[i] - m1[i, j];
            //    }
            //}
            return result;
        }
        public static Matrix operator -(Matrix m1, Vector vector)
        {

            if (vector.Length != m1.Row)
                throw new ArgumentException("向量与矩阵维度不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - vector[i];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] - vector[i]; 
            //    }
            //}
            return result;
        }
        public static Matrix operator -(float scalar, Matrix m1)
        {

            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);


            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalar - m1[i, j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {                 
            //        result[i, j] = scalar - m1[i, j];
            //    }
            //}
            return result;
        }
        public static Matrix operator -(Matrix m1, float scalar)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] - scalar;
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] - scalar;
            //    }
            //}
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

           Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    float temp = 0;


                    for (int k = 0; k < same; k++)
                    {
                        temp += m1[i, k] * m2[k, j];

                    }
                    result[i, j] = temp;
                }
            });

            

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        float temp  = 0;
                   

            //        for (int k = 0; k < same; k++)
            //        {
            //            temp += m1[i, k] * m2[k, j];

            //        }
            //        result[i, j] = temp;
            //    }
            //}
            return result;
        }
        public static float[] operator *(float[] scalars, Matrix m1)
        {

            if (scalars.Length != m1.Row)
                throw new ArgumentException("标量数量与矩阵列数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            float[] result = new float[cols];

            Parallel.For(0, rows, i => {
                float temp = 0;
                for (int j = 0; j < rows; j++)
                {
                    temp += m1[j, i] * scalars[j];
                }
                result[i] = temp;
            });

            return result;
        }

    
        public static Vector operator *(Matrix m1, Vector vector)
        {

            if (vector.Length != m1.Column)
                throw new ArgumentException("向量维度与矩阵行数不一致，无法操作");

            int rows = m1.Row;

            int cols = m1.Column;

            Vector result = new Vector(rows);

            Parallel.For(0, rows, i =>
            {
                float temp = 0;
                for (int j = 0; j < cols; j++)
                {
                    temp += m1[i, j] * vector[j];
                }
                result[i] = temp;
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    float temp = 0;
            //    for (int j = 0; j < cols; j++)
            //    {
            //        temp += m1[i, j] * vector[j];
            //    }
            //    result[i] = temp;
            //}
            return result;
        }
        public static Matrix operator *(float scalar, Matrix m1)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = scalar * m1[i, j];
                }
            });

            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = scalar * m1[i, j]; 
            //    }
            //}
            return result;
        }
        public static Matrix operator *(Matrix m1, float scalar)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m1[i, j] * scalar;
                }
            });
            //for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] * scalar; 
            //    }
            //}
            return result;
        }
        public static Matrix operator /(Matrix m1, float scalar)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
       
                    result[i, j] = scalar == 0?0: m1[i, j] / scalar;
                }
            });

            //    for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        result[i, j] = m1[i, j] / scalar;
            //    }
            //}
            return result;
        }
        public static Matrix operator /(float scalar, Matrix m1)
        {
            int rows = m1.Row;

            int cols = m1.Column;

            Matrix result = new Matrix(rows, cols);

            Parallel.For(0, rows, i => {
                for (int j = 0; j < cols; j++)
                {
                    float value = m1[i, j];
                    result[i, j] = value == 0 ? 0 : scalar / value;
                }
            });

            //    for (int i = 0; i < rows; i++)
            //{
            //    for (int j = 0; j < cols; j++)
            //    {
            //        float value = m1[i, j];
            //        result[i, j] = value == 0 ? 0 : scalar / value;
            //    }
            //}
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
