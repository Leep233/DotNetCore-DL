using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public class ProbabilityDistribution
    {
        /// <summary>
        /// 正态分布
        /// </summary>
        /// <param name="x"></param>
        /// <param name="u"></param>
        /// <param name="a"></param>
        /// <returns></returns>
        public static float NormalDistriution(float x,float u,float a) 
        {
            float a_2 = MathF.Pow(a,2);

            float double_a_2 = 2 * a_2;

           return MathF.Sqrt(1/(double_a_2 * MathF.PI))*MathF.Exp(-(1/double_a_2) * MathF.Pow((x-u), 2));
        }

        /// <summary>
        /// 标准正态分布
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static float StandardNormalDistribution(float x) 
        {
            return MathF.Sqrt(1 / (2 * MathF.PI)) * MathF.Exp(-(1 / 2) * MathF.Pow(x, 2));
        }
    }
}
