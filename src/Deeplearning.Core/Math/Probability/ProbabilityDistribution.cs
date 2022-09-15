﻿using Deeplearning.Core.Math.Common;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Probability
{

    public enum ProbabilityDistributionMode
    {
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
        public static double NormalDistriution(double x, double u, double a)
        {
            double a_2 = MathF.Pow((float)a, 2);

            double double_a_2 = 2 * a_2;

            return MathF.Sqrt((float)(1 / (double_a_2 * MathF.PI))) * MathF.Exp(-((float)(1 / double_a_2)) * MathF.Pow((float)(x - u), 2));
        }

        /// <summary>
        /// 信息量
        /// </summary>
        /// <param name="p">概率</param>
        /// <param name="unit"></param>
        /// <returns></returns>
        public static double Information(double p, InformationUnit unit = InformationUnit.Nats)
        {
            return -LogFunctionByUnit(unit)((float)p);
        }

        /// <summary>
        /// 信息熵
        /// </summary>
        /// <param name="x">参数</param>
        /// <param name="p">参数对应概率</param>
        /// <param name="unit"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double InformationEntropy(double[] x, double[] p, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            double expValue = 0;

            int length = x.Length;

            Func<float, float> logFun = LogFunctionByUnit(unit);

            double scale = ScaleByProbabilityDistributionMode(mode);

            for (int i = 0; i < length; i++)
            {
                double temp = x[i];

                double value = -logFun((float)p[i]);// Information(p[i], unit);// (x[i], unit);

                expValue += temp * value * scale;
            }
            return expValue;
        }

        /// <summary>
        /// 信息熵
        /// </summary>
        /// <param name="x">参数</param>
        /// <param name="p">参数对应概率</param>
        /// <param name="unit"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double InformationEntropy(Vector x, Vector p, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            double expValue = 0;

            int length = x.Length;

            double scale = ScaleByProbabilityDistributionMode(mode);

            Func<float, float> logFun = LogFunctionByUnit(unit);

            for (int i = 0; i < length; i++)
            {
                double temp = x[i];

                double value = -logFun((float)p[i]);// Information(p[i], unit);// (x[i], unit);

                expValue += temp * value * scale;
            }
            return expValue;
        }

        /// <summary>
        /// 信息熵
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="unit"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double InformationEntropy(Func<double[], double[]> probabilityDistributionFunction, double[] x, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            double[] p = probabilityDistributionFunction(x);

            return InformationEntropy(x, p, unit, mode);
        }

        /// <summary>
        /// 信息熵
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="unit"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double InformationEntropy(Func<Vector, Vector> probabilityDistributionFunction, Vector x, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            Vector p = probabilityDistributionFunction(x);

            return InformationEntropy(x, p, unit, mode);
        }



        /// <summary>
        ///  kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static double KL_Divergence(double[] x, double[] p1, double[] p2, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;

            double result = 0;

            var logFunction = LogFunctionByUnit(unit);

            float scale = ScaleByProbabilityDistributionMode(mode);

            for (int i = 0; i < length; i++)
            {

                double temp = logFunction((float)(p1[i])) - logFunction((float)(p2[i]));

                result += x[i] * temp * scale;
            }

            return result;
        }

        /// <summary>
        ///  kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static double KL_Divergence(Vector x, Vector p1, Vector p2, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;

            double result = 0;
 
            var logFunction =  LogFunctionByUnit(unit);
          
            float scale = ScaleByProbabilityDistributionMode(mode);    

            for (int i = 0; i < length; i++)
            {

                double temp = logFunction((float)(p1[i])) - logFunction((float)(p2[i]));

                result += x[i] * temp * scale;           
            }

            return result;
        }

        /// <summary>
        /// kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量y</param>
        /// <param name="p1Function"></param>
        /// <param name="p2Function"></param>
        /// <param name="unit"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double KL_Divergence(Vector x, Func<Vector, Vector> p1Function, Func<Vector, Vector> p2Function, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            Vector p1 = p1Function(x);
            Vector p2 = p2Function(x);

            return KL_Divergence(x, p1, p2, unit, mode);
        }
        /// <summary>
        /// kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量y</param>
        /// <param name="p1Function">分布函数1</param>
        /// <param name="p2Function">分布函数2</param>
        /// <param name="unit"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double KL_Divergence(double[] x, Func<double[], double[]> p1Function, Func<double[], double[]> p2Function, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            double[] p1 = p1Function(x);

            double[] p2 = p2Function(x);

            return KL_Divergence(x, p1, p2, unit, mode);
        }

        private static Func<float, float> LogFunctionByUnit(InformationUnit unit) {
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

        private static float ScaleByProbabilityDistributionMode(ProbabilityDistributionMode mode) {
            float scale = 1;
            switch (mode)
            {
                case ProbabilityDistributionMode.Discrete:
                    scale = 1;
                    break;
                case ProbabilityDistributionMode.Successive:
                    scale = MIN_VALUE;
                    break;
            }
            return scale;
        }

        /// <summary>
        ///  kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static double CrossEntropy(double[] x, double[] p1, double[] p2, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;

            double result = 0;

            double entropy = 0;

            float scale = ScaleByProbabilityDistributionMode(mode);

            Func<float, float> logFunction = LogFunctionByUnit(unit);
           

            for (int i = 0; i < length; i++)
            {
                double p1Value = p1[i];

                double p2Value = p2[i];

                double xValue = x[i];

                double p1LogValue = logFunction((float)p1Value);

                double info = -p1LogValue;

                double temp = p1LogValue - logFunction((float)p2Value);

                result = xValue * temp * scale;

                entropy += xValue * info * scale;
            }

            return entropy + result;
        }

        /// <summary>
        ///  kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static double CrossEntropy(Vector x, Vector p1, Vector p2, InformationUnit unit = InformationUnit.Nats, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;

            double result = 0;

            double entropy = 0;

            float scale = ScaleByProbabilityDistributionMode(mode);

            Func<float, float> logFunction = LogFunctionByUnit(unit);

            for (int i = 0; i < length; i++)
            {
                double p1Value = p1[i];

                double p2Value = p2[i];

                double xValue = x[i];

                double p1LogValue = logFunction((float)p1Value);

                double info = - p1LogValue;

                double temp = p1LogValue - logFunction((float)p2Value);

                result = xValue * temp * scale;

                entropy += xValue * info * scale;
            }

            return entropy + result;
        }



        /// <summary>
        /// 正态分布
        /// </summary>
        /// <param name="x"></param>
        /// <param name="u"></param>
        /// <param name="β"></param>
        /// <returns></returns>
        public static double NormalDistriution(Vector x, Vector u, Matrix β)
        {

            float pi2 = MathF.PI * 2;

            int n = x.Length;

            double det = β.det;

            float f = MathF.Sqrt(((float)det) / MathF.Pow(pi2, n));

            Vector vector = x - u;

            double m = vector.T * β * vector;

            return f * MathF.Exp((float)(m / -2));
        }

        /// <summary>
        /// Laplace分布
        /// </summary>
        /// <param name="x"></param>
        /// <param name="u"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static double Laplace(float x, float u, float y)
        {
            return MathF.Exp(-(MathF.Abs(x - u) / y)) / (2 * y);
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
        public static double Exp(double[] x, double[] p, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = 0;

            float scale = ScaleByProbabilityDistributionMode(mode);

            for (int i = 0; i < length; i++)
            {
                expValue += x[i] * p[i] * scale;
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
        public static double Exp(Vector x, Vector p, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = 0;

            float scale = ScaleByProbabilityDistributionMode(mode);

            for (int i = 0; i < length; i++)
            {
                expValue += x[i] * p[i] * scale;
            }
            return expValue;
        }

        public static double Exp(double[] x, Func<double[], double[]> probabilityDistributionFunction, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            double[] p = probabilityDistributionFunction(x);

            return Exp(x, p, mode);
        }

        public static double Exp(Vector x, Func<Vector, Vector> probabilityDistributionFunction, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            Vector p = probabilityDistributionFunction(x);

            return Exp(x, p, mode);
        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(Vector x, Vector p, ProbabilityDistributionMode mode)
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
        public static double StandardDeviation(Func<double[], double[]> probabilityDistributionFunction, double[] x, ProbabilityDistributionMode mode)
        {
            double[] p = probabilityDistributionFunction(x);

            return MathF.Sqrt((float)Var(x, p, mode));
        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(Func<Vector, Vector> probabilityDistributionFunction, Vector x, ProbabilityDistributionMode mode)
        {
            Vector p = probabilityDistributionFunction(x);

            return MathF.Sqrt((float)Var(x, p, mode));
        }


        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(double[] x, double[] p, ProbabilityDistributionMode mode)
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
        public static double StandardDeviation(double[] x)
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
        public static double Var(double[] x, double[] p, ProbabilityDistributionMode mode)
        {

            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = (float)Exp(x, p, mode);

            double varValue = 0;

            float scale = ScaleByProbabilityDistributionMode(mode);

            for (int i = 0; i < length; i++)
            {
                float val = MathF.Pow((float)(x[i] - expValue), 2);

                varValue += val * p[i] * scale;
            }
            return varValue;

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

            double varValue = 0;

            float scale = ScaleByProbabilityDistributionMode(mode);

            for (int i = 0; i < length; i++)
            {
                float val = MathF.Pow((float)(x[i] - expValue), 2);

                varValue += val * p[i] * scale;
            }
            return varValue;
        }

        /// <summary>
        /// 方差：衡量随机变量x的值差异性
        /// </summary>
        /// <param name="x">所有结果</param>
        /// <param name="p">结果对应发生概率</param>
        /// <param name="mode">变量类型：离散型/连续型</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double Var(Func<Vector, Vector> ProbabilityDistributionFunction, Vector x, ProbabilityDistributionMode mode)
        {
            Vector p = ProbabilityDistributionFunction(x);

            return Var(x, p, mode);
        }


        public static double Var(Func<double[], double[]> ProbabilityDistributionFunction, double[] x, ProbabilityDistributionMode mode)
        {
            double[] p = ProbabilityDistributionFunction(x);


            return Var(x, p, mode);
        }

        public static double Var(double[] x)
        {

            int length = x.Length;

            double avg = Average(x);

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)(x[i] - avg), 2);
            }

            return sum / (length - 1);
        }

        public static double Var(Vector x)
        {
            int length = x.Length;
            double avg = Average(x);
            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)(x[i] - avg), 2);
            }

            return sum / (length - 1);
        }
        public static double Cov(Vector x, Vector y)
        {
            int length = x.Length;

            double xAvg = Average(x);

            double yAvg = Average(y);

            double sum = 0;

            if (length != y.Length) throw new Exception("结果数必须与结果概率数量一致");

            for (int i = 0; i < length; i++)
            {
                sum += (x[i] - xAvg) * (y[i] - yAvg);
            }

            return sum / (length - 1);
        }

        public static double Cov(double[] x, double[] y)
        {
            int length = x.Length;

            double xAvg = Average(x);
            double yAvg = Average(y);
            double sum = 0;
            if (length != y.Length) throw new Exception("结果数必须与结果概率数量一致");
            for (int i = 0; i < length; i++)
            {
                sum += (x[i] - xAvg) * (y[i] - yAvg);
            }

            return sum / (length - 1);
        }

        public static double Average(Vector array)
        {
            int length = array.Length;

            if (length <= 0) return 0;

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += array[i];
            }
            return sum / length;
        }
        public static double Average(double[] array)
        {

            int length = array.Length;

            if (length <= 0) return 0;

            double sum = 0;

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
        public static double Cov(double[] x, double[] y, double[] p, ProbabilityDistributionMode mode = ProbabilityDistributionMode.Discrete)
        {
            int length = x.Length;
            if (length != p.Length || length != y.Length) throw new Exception("结果数必须与结果概率数量一致");

            double xExp = Exp(x, p, mode);

            double yExp = Exp(y, p, mode);

            double covValue = 0;

            float scale = ScaleByProbabilityDistributionMode(mode);

            for (int i = 0; i < length; i++)
            {
                double val = (x[i] - xExp) * (y[i] - yExp);

                covValue += val * p[i] * scale;
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

            double xExp = Exp(x, p, mode);

            double yExp = Exp(y, p, mode);

            double covValue = 0;

            float scale = ScaleByProbabilityDistributionMode(mode);

            for (int i = 0; i < length; i++)
            {
                double val = (x[i] - xExp) * (y[i] - yExp);

                covValue += val * p[i] * scale;
            }
            return covValue;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public static double Bias(Func<double[], double[]> estimator, double[] inputs, double realData) 
        {
          return  (double)(Exp(inputs,estimator(inputs)) - realData);
        }

      
    }
}
