using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Probability
{

    public enum ProbabilityDistributionMode {
        //离散型
        Discrete,
        //连续型
        Successive
    }

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
        /// <summary>
        /// 期望
        /// </summary>
        /// <param name="values"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double Exp(double[] values, ProbabilityDistributionMode mode)
        {
            double average = 0;

            double sum = 0;

            for (int i = 0; i < values.Length; i++) { 
            
                sum +=values[i];
            }

            for (int i = 0; i < values.Length; i++)
            {

            }

            switch (mode)
            {
                case ProbabilityDistributionMode.Discrete:
                    break;
                case ProbabilityDistributionMode.Successive:
                    break;
            }
            return 0;
        }

        /// <summary>
        /// 期望
        /// </summary>
        /// <param name="values"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double Exp(Vector[] values, ProbabilityDistributionMode mode)
        {

            return 0;
        }


       
        

        /// <summary>
        /// 协方差
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static double[] Cov(float[] array) 
        { 
            throw new NotImplementedException();
        }
    }
}
