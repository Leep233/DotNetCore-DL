using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public static class VectorExtension
    {

        /// <summary>
        /// 求均值
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static float Mean(this Vector vector) { 
        
            int length = vector.Length;

            float sum = 0;

            for (int i = 0; i < length; i++) { 
            
                sum += vector[i];
            }

            return sum / length;        
        }

        /// <summary>
        /// 方差
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static float Var(this Vector vector) {
            int length = vector.Length;

            float mean = vector.Mean();

            float sum = 0;

            for (int i = 0; i < length; i++) {

                sum += MathF.Pow((vector[i]-mean),2);
            }
        
            return sum/length;
        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static (float std, float mean) StandardDeviation(this Vector vector)
        {
            int length = vector.Length;

            float mean = vector.Mean();

            float sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)(vector[i] - mean), 2);
            }

            return (MathF.Sqrt((float)(sum / length)),mean);
        }

        public static Vector Normalized(this Vector vector) {
            return vector / Vector.Norm(vector);
        }

        /// <summary>
        /// 最小最大值归一化
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static Vector MinMaxScaler(this Vector vector,int min=0,int max =1) 
        { 
            int length = vector.Length;

            float minScaler = vector[0];

            float maxScaler = minScaler;

            Vector scaleVector = new Vector(length);

            for (int i = 1; i < length; i++)
            {
                float temp = vector[i];

                if(minScaler> temp) minScaler= temp;

                if (maxScaler < temp) maxScaler = temp;
            }

            for (int i = 0; i < length; i++)
            {
                scaleVector[i] = min + (vector[i]-minScaler) / (maxScaler - minScaler) * (max -min);
            }
            return scaleVector;
        }


        /// <summary>
        /// z-score 标准化
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static Vector Standardized(this Vector vector)
        {
            var stdResult = vector.StandardDeviation();

            Vector result = new Vector(vector.Length);

            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = (vector[i] - stdResult.mean) / stdResult.std;
            }

            return result;
        }
    }
}
