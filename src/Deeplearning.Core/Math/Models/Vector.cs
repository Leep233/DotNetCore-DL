using System;
using System.Text;

namespace Deeplearning.Core.Math.Models
{
    public class Vector:ICloneable {

        private float[] scalars;

        public float this[int index] { get => scalars[index]; set => scalars[index] = value; }

        public int Length { get; private set; }

        public virtual float[] T => scalars;// Transpose(scalars);

        public Vector(int length)
        {
            Length = length;
            scalars = new float[Length];
        }

        public Vector(params float [] scalar)
        {
            Length = scalar.Length;
            scalars = scalar;          
        }

        public float Norm(float p)
        {
            float sum = 0;

            switch (p)
            {
                case 1:
                    {
                        for (int i = 0; i < Length; i++)
                        {
                            sum += scalars[i];
                        }
                    }
                    break;
                default:
                    {
                        for (int i = 0; i < Length; i++)
                        {
                            sum += MathF.Pow(scalars[i], p);
                        }
                    }
                    break;
            }


            return MathF.Pow(sum, 1 / p);
        }

        public float MaxNorm()
        {
            float maxValue = 0;

            for (int i = 0; i < Length; i++)
            {
                float temp = MathF.Abs(scalars[i]);
                if (temp > maxValue)
                {
                    maxValue = temp;
                }
            }
            return maxValue;
        }

        public float NoSqrtNorm()
        {
            float sum = 0;
            for (int i = 0; i < Length; i++)
            {
                sum += MathF.Pow(scalars[i], 2);
            }
            return sum;
        }

        public object Clone()
        {
            Vector vector = new Vector(this.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = this[i];
            }

            return vector;
        }

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder("[ ");

            for (int i = 0; i < Length; i++)
            {
                stringBuilder.Append($"{scalars[i].ToString("F4")} ");
            }

            stringBuilder.Append(" ]");

            return stringBuilder.ToString();
        }
      
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
     
        public override bool Equals(object obj)
        {
            Vector v2 = obj as Vector;

            bool result = true;

            int length = this.Length;

            if (v2.Length != length) throw new ArgumentException("维度不一致 无法比较");

            for (int i = 0; i < length; i++)
            {
                if (!this[i].Equals(v2[i]))
                {
                    result = false;
                    break;
                }
            }

            return result;
        }

        public static Vector Random(int length,int min, int max) {

            Vector vector = new Vector(length);

            Random r = new Random(DateTime.Now.GetHashCode());

            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = r.Next(min, max);
            }
            return vector;
        }

        public static Vector Origin(int length) =>new Vector(length);
        
        public static Vector One(int length) {

            Vector vector = new Vector(length);

            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = 1;
            }

            return vector;
        }

        public static Vector Normalize(Vector vector) 
        {
            float m = vector.Norm(2);

            return vector/ m;
        }
        public static Vector Normalized(float[,] col_vector)
        {
            float m = Norm(col_vector,2);

             int rows =  col_vector.GetLength(0);

            Vector v1 = new Vector(rows);

            for (int i = 0; i < rows; i++)
            {
                v1[i] = col_vector[i, 0] / m;
            }

            return v1;
        }
        public static float[] Normalized(float[] row_vector) {

            float m = Norm(row_vector, 2);

            for (int i = 0; i < row_vector.Length; i++)
            {
                row_vector[i] /= m;
            }
            return row_vector;
        }

        public static float[,] Transpose(float[] row_vector)
        {
            float[,] result = new float[row_vector.Length, 1];

            for (int i = 0; i < row_vector.Length; i++)
            {
                result[i, 0] = row_vector[i];
            }
            return result;
        }
        public static float[] Transpose(float[,] col_vector) {

            float[] result = new float[col_vector.Length];

            for (int i = 0; i < col_vector.Length; i++)
            {
                result[i] = col_vector[i, 0];
            }

            return result;
        }

        public static float Norm(float[,] scalars, float p)
        {
            float sum = 0;

            switch (p)
            {
                case 1:
                    {
                        for (int i = 0; i < scalars.Length; i++)
                        {
                            sum += scalars[i,0];
                        }
                    }
                    break;
                default:
                    {
                        for (int i = 0; i < scalars.Length; i++)
                        {
                            sum += MathF.Pow(scalars[i,0], p);
                        }
                    }
                    break;
            }


            return MathF.Pow(sum, 1 / p);
        }
        public static float Norm(float[] scalars,float p)
        {
            float sum = 0;

            switch (p)
            {
                case 1:
                    {
                        for (int i = 0; i < scalars.Length; i++)
                        {
                            sum += scalars[i];
                        }
                    }
                    break;
                default:
                    {
                        for (int i = 0; i < scalars.Length; i++)
                        {
                            sum += MathF.Pow(scalars[i], p);
                        }
                    }
                    break;
            }


            return MathF.Pow(sum, 1 / p);
        }
        public static float MaxNorm(float[] scalars)
        {
            float maxValue = 0;

            for (int i = 0; i < scalars.Length; i++)
            {
                float temp = MathF.Abs(scalars[i]);
                if (temp > maxValue)
                {
                    maxValue = temp;
                }
            }
            return maxValue;
        }
        public static float NoSqrtNorm(float[] scalars)
        {
            float sum = 0;
            for (int i = 0; i < scalars.Length; i++)
            {
                sum += MathF.Pow(scalars[i], 2);
            }
            return sum;
        }

        public static bool IsOrthogonal(Vector v1, Vector v2) 
        {
           return float.MinValue.Equals(v1 * v2);
        }

        public static bool IsOrthogormal(Vector v1, Vector v2) {

            bool result = true;

            if (IsOrthogonal(v1, v2))
            { 
                if (!float.MinValue.Equals(MathF.Abs(1 - v1.Norm(2))))
                {
                    result = false;                   
                }
                else 
                {
                    result = float.MinValue.Equals(MathF.Abs(1 - v2.Norm(2)));                   
                }
            }
            else
            {
                result = false;
            }
            return result;
        }

        public static Matrix operator *(Vector v1, float[] v2) 
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("向量维度不一致");

            int size = v1.Length;

            Matrix matrix = new Matrix(size, size);

            for (int r = 0; r < size; r++)
            {
                for (int c = 0; c < size; c++)
                {
                    matrix[r, c] = v1[r] * v2[c];
                }
            }
            return matrix;

        }
        public static float operator *(float[] v2,Vector v1)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("向量维度不一致");

            float result = 0;

            for (int i = 0; i < v1.Length; i++)
            {
                result += v1[i] * v2[i];
            }
            return (float)result;

        }
        public static float operator *(Vector v1, Vector v2) {
            if (v1.Length != v2.Length)
                throw new ArgumentException("向量维度不一致");

            float result = 0;

            for (int i = 0; i < v1.Length; i++)
            {
                result += v1[i] * v2[i];
            }
            return (float)result;
        }
       
        public static Vector operator *(Vector v1, float scalar)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] * scalar;
            }
            return result;
        }
        public static Vector operator *(float scalar,Vector v1)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] * scalar;
            }
            return result;
        }
       
        public static Vector operator -(Vector v1, Vector v2)
        {
            int size = v1.Length <= v2.Length ? v1.Length : v2.Length;

            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] - v2[i];
            }
            return result;
        }
        public static Vector operator -(Vector v1, float scalar)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] =  v1[i] -scalar;
            }
            return result;
        }
        public static Vector operator -(float scalar, Vector v1)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = scalar- v1[i]  ;
            }
            return result;
        }
       
        public static Vector operator +(Vector v1, Vector v2)
        {
            int size = v1.Length <= v2.Length ? v1.Length : v2.Length;

            Vector result = new Vector(size); 

            for (int i = 0; i < size; i++)
            {
                result[i] = v1[i] + v2[i];
            }
            return result;
        }
        public static Vector operator +(Vector v1, float scalar)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] + scalar;
            }
            return result;
        }
        public static Vector operator +(float scalar, Vector v1)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] + scalar;
            }
            return result;
        }
       
        public static Vector operator /(Vector v1, float scalar)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] / scalar;
            }
            return result;
        }
        public static Vector operator /(float scalar, Vector v1)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = scalar/v1[i]  ;
            }
            return result;
        }
        public static bool operator ==(Vector v1, Vector v2) 
        {
            return v1.Equals(v2);
        }

        public static bool operator !=(Vector v1, Vector v2)
        {
            return !v1.Equals(v2);
        }
    }
}
