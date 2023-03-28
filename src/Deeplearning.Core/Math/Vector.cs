using Deeplearning.Core.Exceptions;
using Deeplearning.Core.Extension;
using System;
using System.Text;

namespace Deeplearning.Core.Math
{

    public struct Vector
    {

        public string Format;

        private float[] scalars;

        public float this[int index] { get => scalars[index]; set => scalars[index] = value; }

        public int Length { get; private set; }

        public float[] T => scalars;

        public Vector(int length)
        {
            Length = length;
            scalars = new float[Length];
            Format = "F4";
        }

        public Vector(params float[] scalar)
        {
            Length = scalar.Length;
            scalars = scalar;
            Format = "F4";
        }

        public static float Norm(Vector vector, float p = 2) 
        {
            float value = 1 / p;
            return MathF.Pow(NoSqrtNorm(vector,p), value);
        }

        public static float NoSqrtNorm(Vector vector,float p=2)
        {
            float sum = 0;

            switch (p)
            {
                case 1:
                    {
                        for (int i = 0; i < vector.Length; i++)
                        {
                            sum += MathF.Abs((float)vector[i]);
                        }
                    }
                    break;
                default:
                    {
                        for (int i = 0; i < vector.Length; i++){

                            float value = MathF.Abs((float)vector[i]);
                            value = MathF.Pow(value, p);
                            sum += value;
                        }
                    }
                    break;
            }
            return sum;
        }

        public static float MaxNorm(Vector vector) 
        {
            float maxValue = vector[0];

            for (int i = 1; i < maxValue; i++)
            {
                float temp = MathF.Abs(vector[i]);

                if (temp > maxValue)
                {
                    maxValue = temp;
                }
            }
            return maxValue;
        }

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder("[ ");

            for (int i = 0; i < Length; i++)
            {
                stringBuilder.Append($"{scalars[i].ToString(Format)} ");
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
            bool result = true;

            if (obj is Vector v2)
            {
                int length = this.Length;

                if (v2.Length != length)
                {
                    result = false;
                }
                else
                {
                    for (int i = 0; i < length; i++)
                    {
                        double value = this[i] - v2[i];

                        if (MathF.Abs((float)value) > MathFExtension.MIN_VALUE)
                        {
                            result = false;
                            break;
                        }
                    }
                }
            }
            else
            {
                result = false;
            }
            return result;
        }

  
        public static Vector Transpose(float[] row_vector)
        {
            Vector result = new Vector(row_vector.Length);

            for (int i = 0; i < row_vector.Length; i++)
            {
                result[i] = row_vector[i];
            }
            return result;
        }

        public static float[] Transpose(Vector vector)
        {
            float[] result = new float[vector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i];
            }
            return result;
        }

        #region 运算符重载

        public static Matrix operator *(Vector vector, float [] array) 
        {
            int size = vector.Length;

            if (size != array.Length)
                throw new OperationException("向量维度不一致");

            Matrix matrix = new Matrix(size, size);

            for (int r = 0; r < size; r++)
            {
                float temp = vector[r];

                if (MathF.Abs(temp) <= MathFExtension.MIN_VALUE) continue;

                for (int c = 0; c < size; c++)
                {
                    matrix[r, c] = temp * array[c];
                }
            }
            return matrix;
        }

        public static float operator * (float[] array,Vector vector)
        {
            if (vector.Length != array.Length)
                throw new OperationException("向量维度不一致");

            float result = 0;

            for (int i = 0; i < vector.Length; i++)
            {
                result += vector[i] * array[i];
            }   
            return result;
        }
        public static float operator *(Vector vector01, Vector vector02) {
            if (vector01.Length != vector02.Length)
                throw new OperationException("向量维度不一致");

            float result = 0;

            for (int i = 0; i < vector01.Length; i++)
            {
                result += vector01[i] * vector02[i];
            }

            return result;
        }     
        public static Vector operator *(Vector vector, float scalar)
        {
            Vector result = new Vector(vector.Length);
    
            for (int i = 0; i < vector.Length; i++) 
            {
                result[i] = vector[i] * scalar;    
            }

            return result;
        }
        public static Vector operator *(float scalar,Vector vector)
        {
            return vector * scalar;
        }
        public static Vector operator -(Vector vector01, Vector vector02)
        {
            int size = (int)MathF.Min(vector01.Length, vector02.Length);

            Vector result = new Vector(size);

            for (int i = 0; i < size; i++)
            {
                result[i] = vector01[i] - vector02[i];
            }

            return result;
        }
        public static Vector operator -(Vector vector, float scalar)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] -= scalar;
            }
            return result;
        }
        public static Vector operator -(float scalar, Vector vector)
        {
            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = scalar - vector[i];
            }
            return result;
        }
        public static Vector operator +(Vector vector01, Vector vector02)
        {
            int size = (int)MathF.Min(vector01.Length, vector02.Length);

            Vector result = new Vector(size);   

            for (int i = 0; i < size; i++)
            {
                result[i] = vector01[i] + vector02[i];
            }
               
            return result;
        }
        public static Vector operator +(Vector vector, float scalar)
        {
            Vector result = new Vector(vector.Length) ;

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] + scalar;
            } 
            return result;
        }
        public static Vector operator +(float scalar, Vector vector)
        {  
            return vector  + scalar;
        }
        public static Vector operator /(Vector vector, float scalar)
        {
            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] / scalar;
            }

            return result;
        }
        public static Vector operator /(float scalar, Vector vector)
        {
            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = scalar / vector[i];
            }
   
            return result;
        }

        public static Vector operator /(Vector v1, Vector v2)
        {
            int length = (int)MathF.Min(v1.Length, v2.Length);

            Vector vector = new Vector(length);

            for (int i = 0; i < length; i++)
            {
                vector[i] = v1[i] / v2[i];
            }
            return vector;
        }

        public static bool operator ==(Vector vector01, Vector vector02) 
        {
            return vector01.Equals(vector02);
        }
        public static bool operator !=(Vector vector01, Vector vector02)
        {
            return !vector01.Equals(vector02);
        }

        #endregion
    }
}
