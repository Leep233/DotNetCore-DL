using System;
using System.Text;

namespace Deeplearning.Core.Math.Models
{
    public class Vector:ICloneable {

        private double[] scalars;

        public double this[int index] { get => scalars[index]; set => scalars[index] = value; }

        public int Length { get; private set; }

        public virtual double[] T => scalars;// Transpose(scalars);

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

        public double Norm(double p)
        {
            double sum = 0;

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
                            sum += MathF.Pow((float)scalars[i],(float) p);
                        }
                    }
                    break;
            }


            return MathF.Pow((float)sum, (float)(1 / p));
        }

        public double MaxNorm()
        {
            double maxValue = 0;

            for (int i = 0; i < Length; i++)
            {
                double temp = MathF.Abs((float)scalars[i]);
                if (temp > maxValue)
                {
                    maxValue = temp;
                }
            }
            return maxValue;
        }

        public double NoSqrtNorm()
        {
            float sum = 0;
            for (int i = 0; i < Length; i++)
            {
                sum += MathF.Pow((float)scalars[i], 2);
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
            double m = vector.Norm(2);

            return vector / m;
        }
        public static Vector Normalized(double[,] col_vector)
        {
            double m = Norm(col_vector,2);

             int rows =  col_vector.GetLength(0);

            Vector v1 = new Vector(rows);

            for (int i = 0; i < rows; i++)
            {
                v1[i] = col_vector[i, 0] / m;
            }

            return v1;
        }
        public static double[] Normalized(double[] row_vector) {

            double m = Norm(row_vector, 2);

            for (int i = 0; i < row_vector.Length; i++)
            {
                row_vector[i] /= m;
            }
            return row_vector;
        }

        public static double[,] Transpose(double[] row_vector)
        {
            double[,] result = new double[row_vector.Length, 1];

            for (int i = 0; i < row_vector.Length; i++)
            {
                result[i, 0] = row_vector[i];
            }
            return result;
        }
        public static double[] Transpose(double[,] col_vector) {

            double[] result = new double[col_vector.Length];

            for (int i = 0; i < col_vector.Length; i++)
            {
                result[i] = col_vector[i, 0];
            }

            return result;
        }

        public static double Norm(double[,] scalars, float p)
        {
            double sum = 0;

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
                            sum += MathF.Pow((float)scalars[i,0],p);
                        }
                    }
                    break;
            }


            return MathF.Pow((float)sum, (float)(1 / p));
        }
        public static double Norm(double[] scalars,float p)
        {
            double sum = 0;

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
                            sum += MathF.Pow((float)scalars[i], p);
                        }
                    }
                    break;
            }


            return MathF.Pow((float)sum, 1 / p);
        }
        public static double MaxNorm(double[] scalars)
        {
            float maxValue = 0;

            for (int i = 0; i < scalars.Length; i++)
            {
                float temp = MathF.Abs((float)scalars[i]);
                if (temp > maxValue)
                {
                    maxValue = temp;
                }
            }
            return maxValue;
        }
        public static double NoSqrtNorm(double[] scalars)
        {
            double sum = 0;
            for (int i = 0; i < scalars.Length; i++)
            {
                sum += MathF.Pow((float)scalars[i], 2);
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
                if (!double.MinValue.Equals(MathF.Abs((float)(1 - v1.Norm(2)))))
                {
                    result = false;                   
                }
                else 
                {
                    result = double.MinValue.Equals(MathF.Abs((float)(1 - v2.Norm(2))));                   
                }
            }
            else
            {
                result = false;
            }
            return result;
        }

        public static Matrix operator *(Vector v1, double[] v2) 
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
        public static double operator *(double[] v2,Vector v1)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("向量维度不一致");

            double result = 0;

            for (int i = 0; i < v1.Length; i++)
            {
                result += v1[i] * v2[i];
            }
            return (double)result;

        }
        public static double operator *(Vector v1, Vector v2) {
            if (v1.Length != v2.Length)
                throw new ArgumentException("向量维度不一致");

            double result = 0;

            for (int i = 0; i < v1.Length; i++)
            {
                result += v1[i] * v2[i];
            }
            return result;
        }
       
        public static Vector operator *(Vector v1, double scalar)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] * scalar;
            }
            return result;
        }
        public static Vector operator *(double scalar,Vector v1)
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
        public static Vector operator -(Vector v1, double scalar)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] =  v1[i] -scalar;
            }
            return result;
        }
        public static Vector operator -(double scalar, Vector v1)
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
        public static Vector operator +(Vector v1, double scalar)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] + scalar;
            }
            return result;
        }
        public static Vector operator +(double scalar, Vector v1)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] + scalar;
            }
            return result;
        }
       
        public static Vector operator /(Vector v1, double scalar)
        {
            Vector result = new Vector(v1.Length);

            for (int i = 0; i < v1.Length; i++)
            {
                result[i] = v1[i] / scalar;
            }
            return result;
        }
        public static Vector operator /(double scalar, Vector v1)
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
