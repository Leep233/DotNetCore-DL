using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Probability
{

    public enum ProbabilityDistributionMode {
        /// <summary>
        /// 离散型
        /// </summary>
        Discrete,
        /// <summary>
        /// 连续型
        /// </summary>
        Successive
    }

    public class ProbabilityDistribution
    {

        public const float MIN_VALUE = 1E-3F;

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
        /// <param name="x">所有结果</param>
        /// <param name="p">结果对应发生概率</param>
        /// <param name="mode">变量类型：离散型/连续型</param>
        /// <returns>期望值</returns>
        /// <exception cref="Exception"></exception>
        public static double Exp(float [] x, float[] p, ProbabilityDistributionMode mode= ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = 0;

            for (int i = 0; i < length; i++)
            {
                switch (mode)
                {
                    case ProbabilityDistributionMode.Discrete:
                        expValue += x[i] * p[i];
                        break;
                    case ProbabilityDistributionMode.Successive:
                        expValue += x[i] * p[i] * MIN_VALUE;
                        break;
                }
            }

            
            return expValue;
        }

        /// <summary>
        /// 期望 所有对应结果与对应概率发生的和
        /// </summary>
        /// <param name="x">所有结果</param>
        /// <param name="p">结果对应发生概率</param>
        /// <param name="mode">变量类型：离散型/连续型</param>
        /// <returns>期望值</returns>
        /// <exception cref="Exception"></exception>
        public static double Exp(Vector x,Vector p ,ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = 0;

            for (int i = 0; i < length; i++)
            {
                switch (mode)
                {
                    case ProbabilityDistributionMode.Discrete:
                        expValue += x[i] * p[i];
                        break;
                    case ProbabilityDistributionMode.Successive:
                        expValue += x[i] * p[i] * MIN_VALUE;
                        break;
                }
            }
            return expValue;
        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(Vector x, Vector p, ProbabilityDistributionMode mode) {

            return MathF.Sqrt((float)Var(x, p, mode));
        
        }
        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(float[] x, float[] p, ProbabilityDistributionMode mode)
        {

            return MathF.Sqrt((float)Var(x, p, mode));

        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(float[] x)
        {

            return MathF.Sqrt((float)Var(x));

        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(Vector x) 
        {
            return MathF.Sqrt((float)Var(x));

        }

        /// <summary>
        ///  方差：衡量随机变量x的值差异性
        /// </summary>
        /// <param name="x">所有结果</param>
        /// <param name="p">结果对应发生概率</param>
        /// <param name="mode">变量类型：离散型/连续型</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double Var(float[] x,float[]p, ProbabilityDistributionMode mode) {

            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            float expValue = (float)Exp(x,p,mode);

            double xSum = 0;

            for (int i = 0; i < length; i++)
            {
                float val = MathF.Pow(x[i] - expValue, 2);

                switch (mode)
                {
                    case ProbabilityDistributionMode.Discrete:
                        xSum += val * p[i];
                        break;
                    case ProbabilityDistributionMode.Successive:
                        xSum += val * p[i] * MIN_VALUE;
                        break;
                }        
            }

            return xSum;

        }

        /// <summary>
        /// 方差：衡量随机变量x的值差异性
        /// </summary>
        /// <param name="x">所有结果</param>
        /// <param name="p">结果对应发生概率</param>
        /// <param name="mode">变量类型：离散型/连续型</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double Var(Vector x, Vector p, ProbabilityDistributionMode mode)
        {

            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            float expValue = (float)Exp(x, p, mode);

            double xSum = 0;

            for (int i = 0; i < length; i++)
            {
                float val = MathF.Pow(x[i] - expValue, 2);

                switch (mode)
                {
                    case ProbabilityDistributionMode.Discrete:
                        xSum += val * p[i];
                        break;
                    case ProbabilityDistributionMode.Successive:
                        xSum += val * p[i] * MIN_VALUE;
                        break;
                }
            }

            return xSum;

        }

        public static double Var(float[] x) { 
            int length = x.Length;
            float avg = Average(x);
            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)x[i] - avg, 2);
            }

            return sum/(length-1);
        }

        public static double Var(Vector x)
        {
            int length = x.Length;
            float avg = Average(x);
            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)x[i] - avg, 2);
            }

            return sum / (length - 1);
        }
        public static double Cov(Vector x, Vector y)
        {
            int length = x.Length;

            float xAvg = Average(x);
            float yAvg = Average(y);
            double sum = 0;
            if (length != y.Length) throw new Exception("结果数必须与结果概率数量一致");
            for (int i = 0; i < length; i++)
            {
                sum += ((float)x[i] - xAvg) * ((float)y[i] - yAvg);
            }

            return sum / (length - 1);
        }

        public static double Cov(float[] x, float[] y) 
        {
            int length = x.Length;

            float xAvg = Average(x);
            float yAvg = Average(y);
            double sum = 0;
            if (length != y.Length) throw new Exception("结果数必须与结果概率数量一致");
            for (int i = 0; i < length; i++)
            {
                sum += ((float)x[i] - xAvg) * ((float)y[i] - yAvg);
            }

            return sum / (length - 1);
        }

        public static float Average(Vector array)
        {

            int length = array.Length;

            if (length <= 0) return 0;

            float sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += array[i];
            }
            return sum / length;
        }
        public static float Average(float[] array) {
        
            int length = array.Length;

            if(length<=0)return 0;

            float sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += array[i];
            }
            return sum / length;
        }

        /// <summary>
        /// 协方差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns>协方差>0正相关，<0 负相关 =0不相关</returns>
        /// <exception cref="Exception"></exception>
        public static double Cov(float[] x, float[] y,float[]p, ProbabilityDistributionMode mode= ProbabilityDistributionMode.Discrete) 
        {
            int length = x.Length;
            if (length != p.Length || length != y.Length) throw new Exception("结果数必须与结果概率数量一致");
       

            float xExp = (float)Exp(x, p, mode);
            float yExp = (float)Exp(y, p, mode);

            double covValue = 0;

            for (int i = 0; i < length; i++)
            {
                float val = (x[i] - xExp)*(y[i]-yExp);

                switch (mode)
                {
                    case ProbabilityDistributionMode.Discrete:
                        covValue += val * p[i];
                        break;
                    case ProbabilityDistributionMode.Successive:
                        covValue += val * p[i] * MIN_VALUE;
                        break;
                }
            }

            return covValue;

        }
        /// <summary>
        /// 协方差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns>协方差>0正相关，<0 负相关 =0不相关</returns>
        /// <exception cref="Exception"></exception>
        public static double Cov(Vector x, Vector y, Vector p, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;
            if (length != p.Length || length != y.Length) throw new Exception("结果数必须与结果概率数量一致");


            float xExp = (float)Exp(x, p, mode);
            float yExp = (float)Exp(y, p, mode);

            double covValue = 0;

            for (int i = 0; i < length; i++)
            {
                float val = (x[i] - xExp) * (y[i] - yExp);

                switch (mode)
                {
                    case ProbabilityDistributionMode.Discrete:
                        covValue += val * p[i];
                        break;
                    case ProbabilityDistributionMode.Successive:
                        covValue += val * p[i] * MIN_VALUE;
                        break;
                }
            }

            return covValue;

        }
   
    
        
    }
}
