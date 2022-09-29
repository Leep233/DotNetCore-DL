using Deeplearning.Core.Exceptions;
using Deeplearning.Core.Extension;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math.Models
{
    public struct Vector
    {

        private double[] scalars;

        public double this[int index] { get => scalars[index]; set => scalars[index] = value; }

        public int Length { get; private set; }

        public double[] T => scalars;

        public Vector(int length)
        {
            Length = length;
            scalars = new double[Length];
        }

        public Vector(params double[] scalar)
        {
            Length = scalar.Length;
            scalars = scalar;          
        }

        public static double Norm(Vector vector, float p = 2) 
        {
            float value = 1 / p;
            return MathF.Pow((float)NoSqrtNorm(vector,p), value);
        }

        public static double NoSqrtNorm(Vector vector,float p=2)
        {
            double sum = 0;

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

        public static double MaxNorm(Vector vector) 
        {
            double maxValue = vector[0];

            for (int i = 1; i < maxValue; i++)
            {
                double temp = MathF.Abs((float)vector[i]);

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

                        if (MathF.Abs((float)value) > 10E-15)
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

        public static Vector Random(int length,int min, int max) {

            Vector vector = new Vector(length);

            Random r = new Random(DateTime.Now.GetHashCode());

            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = r.Next(min, max);
            }
            return vector;
        }

        public static Vector One(int length) {

            Vector vector = new Vector(length);

            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = 1;
            }

            return vector;
        }

        public static Vector Normalized(Vector vector) {

            double minValue = vector[0];

            double maxValue = minValue;

            for (int i = 1; i < vector.Length; i++)
            {
                double temp = vector[i];

                if (temp > minValue)
                {
                    maxValue = temp;
                }
                if (temp < minValue)
                {
                    minValue = temp;
                }
            }
            return (vector - minValue) / (maxValue - minValue);
        }

        /// <summary>
        /// 向量标准化
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Vector Standardized(Vector vector) 
        {
            double m = Vector.Norm(vector,2);

            if (m == 0) throw new InvalidOperationException("被除数不能为0");

            return vector / m;
        }
     
        public static Vector Transpose(double[] row_vector)
        {
            Vector result = new Vector(row_vector.Length);

            for (int i = 0; i < row_vector.Length; i++)
            {
                result[i] = row_vector[i];
            }
            return result;
        }

        public static double[] Transpose(Vector vector)
        {

            double[] result = new double[vector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i];
            }

            return result;
        }


        public static Matrix operator *(Vector vector, double [] array) 
        {
            int size = vector.Length;

            if (size != array.Length)
                throw new OperationException("向量维度不一致");

            Matrix matrix = new Matrix(size, size);

            for (int r = 0; r < size; r++)
            {
                double temp = vector[r];

                for (int c = 0; c < size; c++)
                {
                    matrix[r, c] = temp * array[c];
                }
            }
            return matrix;
        }
        public static double operator *(double[] array,Vector vector)
        {
            if (vector.Length != array.Length)
                throw new OperationException("向量维度不一致");

            double result = 0;

            for (int i = 0; i < vector.Length; i++)
            {
                result += vector[i] * array[i];
            }   
            return result;
        }
        public static double operator *(Vector vector01, Vector vector02) {
            if (vector01.Length != vector02.Length)
                throw new OperationException("向量维度不一致");

            double result = 0;

            for (int i = 0; i < vector01.Length; i++)
            {
                result += vector01[i] * vector02[i];
            }

            return result;
        }     
        public static Vector operator *(Vector vector, double scalar)
        {
            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++) {
                result[i] = vector[i] * scalar;     
            }

            return result;
        }
        public static Vector operator *(double scalar,Vector vector)
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
        public static Vector operator -(Vector vector, double scalar)
        {
            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] - scalar;
            }
            return result;
        }
        public static Vector operator -(double scalar, Vector vector)
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
        public static Vector operator +(Vector vector, double scalar)
        {
            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] + scalar;
            } 
            return result;
        }
        public static Vector operator +(double scalar, Vector vector)
        {  
            return vector+ scalar;
        }
        public static Vector operator /(Vector vector, double scalar)
        {
            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                double value = scalar == 0 ? 0 : vector[i] / scalar;

                result[i] = value;
            }

            return result;
        }
        public static Vector operator /(double scalar, Vector vector)
        {
            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                double value = vector[i];
                value = (value == 0 ? 0 : scalar / value);
                result[i] = value;
            }
   
            return result;
        }
        public static bool operator ==(Vector vector01, Vector vector02) 
        {
            return vector01.Equals(vector02);
        }
        public static bool operator !=(Vector vector01, Vector vector02)
        {
            return !vector01.Equals(vector02);
        }
    }
}
