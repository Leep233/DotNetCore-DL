using Deeplearning.Core.Extension;
using Deeplearning.Core.Math.Probability;
using System;

namespace Deeplearning.Core.Math.Common
{
    public static class CommonFunctions
    {
        public static float Sigmoid(float x)
        {
            return 1 / (1 + MathF.Exp(-x));
        }

        public static Vector Sigmoid(Vector vector)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = Sigmoid((float)vector[i]);
            }
            return vector;
        }

        public static Matrix Sigmoid(Matrix matrix) 
        {
            Matrix m = matrix;

            for (int i = 0; i < m.Row; i++)
            {
                for (int j = 0; j < m.Column; j++)
                {
                    m[i, j] = Sigmoid((float)m[i, j]);
                }
            }

            return m;
        }

        public static Vector Softplus(Vector vector)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = Softplus((float)vector[i]);
            }
            return vector;
        }



        public static Matrix Softplus(Matrix matrix) { 
        
            Matrix m = matrix;

            for (int i = 0; i < m.Row; i++)
            {
                for (int j = 0; j < m.Column; j++)
                {
                    m[i,j] = Softplus((float)m[i, j]);
                }
            }
            return m;        
        }

        public static float Softplus(float x) { 
            return MathF.Log(1+ MathF.Exp(x));
        }


        public static Vector Softmax(Vector vector) {

            double maxvalue = vector[0];

            for (int i = 1; i < vector.Length; i++)
            {
                double temp = vector[i];

                if (temp > maxvalue) maxvalue = temp;
            }

            double sum = 0;

            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                double exp = MathF.Exp((float)(vector[i] - maxvalue));

                result[i] = exp;

                sum += exp;
            }

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] /= sum;
            }
            return result;
        }

        public static Matrix Softmax(Matrix matrix) {

            Matrix m = matrix;

            double MaxValue = MathFExtension.Max(m);

            double sum = 0;

            for (int i = 0; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    double exp = MathF.Exp((float)(m[i, j] - MaxValue));

                    m[i, j] = exp;

                    sum += exp;
                }
            }

            for (int i = 0; i < m.Row; i++)
            {
                for (int j = 0; j < m.Column; j++)
                {
                    m[i, j] /= sum;
                }
            }

            return m;
        }

        /// <summary>
        /// 通过信息量的单位 来选择对应的对数函数
        /// </summary>
        /// <param name="unit">信息量单位</param>
        /// <returns>对数函数</returns>
        public static Func<float, float> MathfLog(InformationUnit unit)
        {
            Func<float, float> logFunction = MathF.Log;

            switch (unit)
            {
                case InformationUnit.Bits:
                    logFunction = MathF.Log2;
                    break;
                case InformationUnit.Hart:
                    logFunction = MathF.Log10;
                    break;
                case InformationUnit.Nats:
                default:
                    logFunction = MathF.Log;
                    break;
            }
            return logFunction;
        }

        /// <summary>
        /// 根据随机变量类型 来选择概率分布的收缩率
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static float ProbabilityDistributionScale(RandomVariableType type)
        {
            float scale = 1;

            switch (type)
            {
                case RandomVariableType.Discrete:
                    scale = 1;
                    break;
                case RandomVariableType.Successive:
                    scale = 0.001F;
                    break;
            }
            return scale;
        }

        public static Vector Tanh(Vector vector)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = Tanh((float)vector[i]);               
            }
            return vector;
        }

        public static Matrix Tanh(Matrix matrix) {

            for (int i = 0; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    matrix[i, j] = Tanh((float)matrix[i, j]);
                }
            }
            return matrix;
        }
       
        public static float Tanh(float x) {

            float e1 = MathF.Pow(MathF.E,x);

            float e2 = MathF.Pow(MathF.E,-x);

            return (e1 - e2) / (e1 + e2);
        }

        public static Matrix ReLU(Matrix matrix) {

            for (int i = 0; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    matrix[i, j] = MathF.Max(0, (float)matrix[i, j]);
                }
            }
            return matrix;
        }
    
    }
}
