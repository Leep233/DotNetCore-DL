using System;
using System.Text;
using System.Threading.Tasks;

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

        public Vector(params float[] scalar)
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
                        Parallel.For(0, Length, i => {
                            sum += scalars[i];
                        });
                        //for (int i = 0; i < Length; i++)
                        //{
                        //    sum += scalars[i];
                        //}
                    }
                    break;
                default:
                    {

                        Parallel.For(0, Length, i => {
                            float value = MathF.Pow((float)scalars[i], (float)p);
                            sum += value;
                        });

                        //for (int i = 0; i < Length; i++)
                        //{
                        //    float value = MathF.Pow((float)scalars[i], (float)p);
                        //    sum += value;                     
                        //}
                    }
                    break;
            }


            return MathF.Pow((float)sum, (float)(1 / p));
        }

        public double MaxNorm()
        {
            double maxValue = 0;

            Parallel.For(0, Length, i => {
                double temp = Validator.ZeroValidation(MathF.Abs((float)scalars[i]));

                if (temp > maxValue)
                {
                    maxValue = temp;
                }
            });

            //for (int i = 0; i < Length; i++)
            //{
            //    double temp = Validator.ZeroValidation(MathF.Abs((float)scalars[i]));

            //    if (temp > maxValue)
            //    {
            //        maxValue = temp;
            //    }
            //}
            return maxValue;
        }

        public double NoSqrtNorm()
        {
            double sum = 0;

            Parallel.For(0, Length, i => {
                sum += Validator.ZeroValidation(MathF.Pow((float)scalars[i], 2));
            });
            //for (int i = 0; i < Length; i++)
            //{
            //    sum += Validator.ZeroValidation(MathF.Pow((float)scalars[i], 2));
            //}
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
                if (Validator.ZeroValidation(this[i] - v2[i])!=0)
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

        public static Vector Normalized(Vector vector) { 

            float minValue = vector[0];

            float maxValue = minValue;

            Parallel.For(1, vector.Length, i =>
            {
                float temp = vector[i];

                if (temp > minValue)
                {
                    maxValue = temp;
                }
                if (temp < minValue)
                {
                    minValue = temp;
                }

            });

            //for (int i = 1; i < vector.Length; i++)
            //{
            //    float temp = vector[i];

            //    if (temp > minValue) {
            //        maxValue = temp;
            //    }
            //    if (temp < minValue)
            //    {
            //        minValue = temp;
            //    }
            //}

            return (vector - minValue) / (maxValue - minValue);

        }

        public static Vector Standardized(Vector vector) 
        {
            float m = vector.Norm(2);

            return  Validator.ZeroValidation(m)==0 ? new Vector(vector.Length) : vector / m;
     
        }
        public static float Min(float[] v)
        {
            float value = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (value > v[i]) value = v[i];
            }
            return value;
        }
        public static float Max(float[] v)
        {
            float value = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (value < v[i]) value = v[i];
            }
            return value;
        }
        public static float Min(Vector v)
        {
            float value = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (value > v[i]) value = v[i];
            }
            return value;
        }
        public static float Max(Vector v)
        {
            float value = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (value < v[i]) value = v[i];
            }
            return value;
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
            double sum = 0;

            switch (p)
            {
                case 1:
                    {
                        Parallel.For(0, scalars.GetLength(0), i=>{ sum += scalars[i, 0]; });

                        //for (int i = 0; i < scalars.Length; i++)
                        //{
                        //    sum += scalars[i,0];
                        //}
                    }
                    break;
                default:
                    {

                        Parallel.For(0, scalars.GetLength(0), i => { sum += MathF.Pow((float)scalars[i, 0], p); });

                        //for (int i = 0; i < scalars.Length; i++)
                        //{
                        //    sum += MathF.Pow((float)scalars[i,0],p);
                        //}
                    }
                    break;
            }


            return MathF.Pow((float)sum, (float)(1 / p));
        }
        public static float Norm(float[] scalars,float p)
        {
            float sum = 0;

            switch (p)
            {
                case 1:
                    {

                        Parallel.For(0, scalars.Length, i => { sum += scalars[i]; });

                        //for (int i = 0; i < scalars.Length; i++)
                        //{
                        //    sum += scalars[i];
                        //}
                    }
                    break;
                default:
                    {

                        Parallel.For(0, scalars.Length, i => { sum += MathF.Pow(scalars[i], p); });

                        //for (int i = 0; i < scalars.Length; i++)
                        //{
                        //    sum += MathF.Pow((float)scalars[i], p);
                        //}
                    }
                    break;
            }


            return MathF.Pow((float)sum, 1 / p);
        }
        public static float MaxNorm(float[] scalars)
        {
            float maxValue = 0;
            Parallel.For(0, scalars.Length, i => {
                float temp = MathF.Abs((float)scalars[i]);

                maxValue = MathF.Max(maxValue, temp);
            });
            //for (int i = 0; i < scalars.Length; i++)
            //{
            //    float temp = MathF.Abs((float)scalars[i]);

            //    maxValue = MathF.Max(maxValue, temp);
            //}
            return maxValue;
        }
        public static float NoSqrtNorm(float[] scalars)
        {
            float sum = 0;
            Parallel.For(0, scalars.Length, i => {

                sum += MathF.Pow((float)scalars[i], 2);
            });
            //    for (int i = 0; i < scalars.Length; i++)
            //{
            //    sum += MathF.Pow((float)scalars[i], 2);
            //}
            return sum;
        }


        public static Matrix operator *(Vector v1, float[] v2) 
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("向量维度不一致");

            int size = v1.Length;

            Matrix matrix = new Matrix(size, size);

            Parallel.For(0, size, r => {
                for (int c = 0; c < size; c++)
                {
                    matrix[r, c] = v1[r] * v2[c];
                }
            });

            return matrix;
        }

        public static float operator *(float[] v2,Vector v1)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("向量维度不一致");

            float result = 0;


            Parallel.For(0, v1.Length, i => 
            {
                result += v1[i] * v2[i];
            });

            return Validator.ZeroValidation(result);

        }
        public static float operator *(Vector v1, Vector v2) {
            if (v1.Length != v2.Length)
                throw new ArgumentException("向量维度不一致");

            float result = 0;

            Parallel.For(0, v1.Length, i =>
            {
                result += v1[i] * v2[i];
            });

            return Validator.ZeroValidation( result);
        }
       
        public static Vector operator *(Vector v1, float scalar)
        {
            Vector result = new Vector(v1.Length);


            Parallel.For(0, v1.Length, i =>
            {
                result[i] = v1[i] * scalar;
            });

            return result;
        }
        public static Vector operator *(float scalar,Vector v1)
        {
            return v1* scalar;
        }
         

        public static Vector operator -(Vector v1, Vector v2)
        {

            int size = (int)MathF.Min(v1.Length, v2.Length);

            Vector result = new Vector(size);


            Parallel.For(0, v1.Length, i =>
            {
                result[i] = v1[i] - v2[i];
            });

            return result;
        }
        public static Vector operator -(Vector v1, float scalar)
        {
            Vector result = new Vector(v1.Length);

            Parallel.For(0, v1.Length, i => {
                result[i] = v1[i] - scalar;
            });
            return result;
        }
        public static Vector operator -(float scalar, Vector v1)
        {
            Vector result = new Vector(v1.Length);
            Parallel.For(0, v1.Length, i => {
                result[i] = scalar - v1[i];
            });

            return result;
        }
       
        public static Vector operator +(Vector v1, Vector v2)
        {
            int size = (int)MathF.Min(v1.Length, v2.Length);

            Vector result = new Vector(size);
            Parallel.For(0, size, i => {
                result[i] = v1[i] + v2[i];
            });
   
            return result;
        }
        public static Vector operator +(Vector v1, float scalar)
        {
            Vector result = new Vector(v1.Length);


            Parallel.For(0, v1.Length, i => {
                result[i] = v1[i] + scalar;
            });
            return result;
        }
        public static Vector operator +(float scalar, Vector v1)
        {  
            return v1+ scalar;
        }
       
        public static Vector operator /(Vector v1, float scalar)
        {
            Vector result = new Vector(v1.Length);

            Parallel.For(0, v1.Length, i => {
                result[i] = scalar == 0 ? 0 : v1[i] / scalar;
            });
  
            return result;
        }
        public static Vector operator /(float scalar, Vector v1)
        {
            Vector result = new Vector(v1.Length);

            Parallel.For(0, v1.Length, i => {
                float value = v1[i];

                result[i] = value == 0 ? 0 : scalar / value;
            });
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
