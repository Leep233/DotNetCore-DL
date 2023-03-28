using Deeplearning.Core.Extension;
using Deeplearning.Core.Math.Common;
using System;

namespace Deeplearning.Core.Math.Probability
{
    /// <summary>
    /// 随机变量类型
    /// </summary>
    public enum RandomVariableType
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

    public class Distribution
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

            double det = Matrix.Det(β);//.det;

            float f = MathF.Sqrt(((float)det) / MathF.Pow(pi2, n));

            Vector vector = x - u;

            double m = vector.T * Matrix.Dot(β , vector);

            return f * MathF.Exp((float)(m / -2));
        }


        /// <summary>
        /// 正态分布
        /// </summary>
        /// <param name="x"></param>
        /// <param name="u"></param>
        /// <param name="β"></param>
        /// <returns></returns>
        public static Vector NormalDistriution(Matrix x, Matrix u, Matrix β)
        {
            float pi2 = MathF.PI * 2;

            int count = x.Row;

            int n = x.Column;

            double det = Matrix.Det(β);//.det;

            float f = MathF.Sqrt(((float)det) / MathF.Pow(pi2, n));

            Vector result = new Vector(count);

            Vector vector = new Vector(n);

            for (int i = 0; i < count; i++)
            {

                for (int j = 0; j < n; j++)
                {
                    vector[j] = x[i, j] - u[i, j];
                }

                double m = vector.T * Matrix.Dot(β, vector);

                result[i] = f* MathF.Exp((float)(m / -2));
            }

            return result;
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

        #region 期望函数

        /// <summary>
        ///   期望：该概率分布最有可能出现的值（所有随机变量与对应概率分布概率乘积的和）
        /// </summary>
        /// <param name="x">随机变量</param>
        /// <param name="p">概率分布概率</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns>期望值</returns>
        /// <exception cref="Exception"></exception>
        public static double Exp(float[] x, float[] p, RandomVariableType mode = RandomVariableType.Discrete)
        {
            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(mode);

            for (int i = 0; i < length; i++)
            {
                expValue += x[i] * p[i] * scale;
            }

            return expValue;
        }

        /// <summary>
        ///   期望：该概率分布最有可能出现的值（所有随机变量与对应概率分布概率乘积的和）
        /// </summary>
        /// <param name="x">随机变量</param>
        /// <param name="p">概率分布概率</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns>期望值</returns>
        /// <exception cref="Exception"></exception>
        public static double Exp(double[] x, double[] p, RandomVariableType mode = RandomVariableType.Discrete)
        {
            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(mode);

            for (int i = 0; i < length; i++)
            {
                expValue += x[i] * p[i] * scale;
            }

            return expValue;
        }

        /// <summary>
        ///   期望：该概率分布最有可能出现的值（所有随机变量与对应概率分布概率乘积的和）
        /// </summary>
        /// <param name="x">随机变量</param>
        /// <param name="p">概率分布概率</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns>期望值</returns>
        /// <exception cref="Exception"></exception>
        public static double Exp(Vector x, Vector p, RandomVariableType type = RandomVariableType.Discrete)
        {
            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                expValue += x[i] * p[i] * scale;
            }
            return expValue;
        }

        /// <summary>
        /// 期望：该概率分布最有可能出现的值（所有随机变量与对应概率分布概率乘积的和）
        /// </summary>
        /// <typeparam name="T">支持类型：double[] ; float[] ; Vector ;</typeparam>
        /// <typeparam name="T1">支持类型：double[] ; float[] ; Vector ;</typeparam>
        /// <param name="x">随机变量</param>
        /// <param name="pdf">概率分布函数</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns>期望值</returns>
        /// <exception cref="ArgumentException"></exception>
        public static double Exp<T>(T x, Func<T, T> pdf, RandomVariableType type = RandomVariableType.Discrete)
        {
            T pd = pdf(x);

            double result = 0;

            if (x is double[] xdArray && pd is double[] pddArray)
            {
                result = Exp(xdArray, pddArray, type);
            }
            else if (x is float[] xfArray && pd is float[] pdfArray)
            {
                result = Exp(xfArray, pdfArray, type);
            }
            else if (x is Vector xv && pd is Vector pdv)
            {
                result = Exp(xv, pdv, type);
            }
            else
            {
                throw new ArgumentException("This parameter is not supported!!(support:float[] double[],Vector)");
            }
            return result;
        
        }

        #endregion

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="pd"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(Vector x, Vector pd, RandomVariableType mode)
        {
            return MathF.Sqrt((float)Var(x, pd, mode));
        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(Func<double[], double[]> pdf, double[] x, RandomVariableType mode)
        {
            double[] p = pdf(x);

            return MathF.Sqrt((float)Var(x, p, mode));
        }

        /// <summary>
        /// 标准差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double StandardDeviation(Func<Vector, Vector> probabilityDistributionFunction, Vector x, RandomVariableType mode)
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
        public static double StandardDeviation(double[] x, double[] p, RandomVariableType mode)
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
        ///  方差：衡量随机变量x的值差异性((随机变量-期望)^2的和)
        /// </summary>
        /// <param name="x">随机变量</param>
        /// <param name="pd">概率分布概率</param>
        /// <param name="type">变量类型：离散型/连续型</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double Var(float[] x, float[] pd, RandomVariableType type)
        {

            int length = x.Length;

            if (length != pd.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = Exp(x, pd, type);

            double varValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                float val = MathF.Pow((float)(x[i] - expValue), 2);

                varValue += val * pd[i] * scale;
            }
            return varValue;
        }

        /// <summary>
        ///  方差：衡量随机变量x的值差异性((随机变量-期望)^2的和)
        /// </summary>
        /// <param name="x">随机变量</param>
        /// <param name="pd">概率分布概率</param>
        /// <param name="type">变量类型：离散型/连续型</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double Var(double[] x, double[] pd, RandomVariableType type = RandomVariableType.Discrete)
        {
            int length = x.Length;

            if (length != pd.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = Exp(x, pd, type);

            double varValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                float val = MathF.Pow((float)(x[i] - expValue), 2);

                varValue += val * pd[i] * scale;
            }
            return varValue;
        }

        /// <summary>
        ///  方差：衡量随机变量x的值差异性((随机变量-期望)^2的和)
        /// </summary>
        /// <param name="x">随机变量</param>
        /// <param name="pd">概率分布概率</param>
        /// <param name="type">变量类型：离散型/连续型</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double Var(Vector x, Vector p, RandomVariableType mode)
        {

            int length = x.Length;

            if (length != p.Length) throw new Exception("结果数必须与结果概率数量一致");

            double expValue = Exp(x, p, mode);

            double varValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(mode);

            for (int i = 0; i < length; i++)
            {
                float val = MathF.Pow((float)(x[i] - expValue), 2);

                varValue += val * p[i] * scale;
            }
            return varValue;
        }

        /// <summary>
        /// 方差：衡量随机变量x的值差异性((随机变量-期望)^2的和)
        /// </summary>
        /// <typeparam name="T">支持类型：double[] ; float[] ; Vector ;</typeparam>
        /// <param name="x">随机变量</param>
        /// <param name="pdf">概率分布函数</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static double Var<T>(T x, Func<T, T> pdf,RandomVariableType type)
        {
            T pd = pdf(x);

            double result = 0;

            if (x is double[] x_d_value && pd is double[] pd_d_value)
            {
                result = Var(x_d_value, pd_d_value, type);
            }
            else if (x is float[] x_f_value && pd is float[] pd_f_value)
            {
                result = Var(x_f_value, pd_f_value, type);
            }
            else if (x is Vector x_v_value && pd is Vector pd_v_value)
            {
                result = Var(x_v_value, pd_v_value, type);
            }
            else
            {
                throw new ArgumentException("This parameter is not supported!!(support:float[] double[],Vector)");
            }
            return result;
        }

        /// <summary>
        /// 均值方差
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Var(float[] x)
        {

            int length = x.Length;

            double avg = MathFExtension.Average(x);

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)(x[i] - avg), 2);
            }

            return sum / (length - 1);
        }

        /// <summary>
        /// 均值方差
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Var(double[] x)
        {

            int length = x.Length;

            double avg = MathFExtension.Average(x);

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)(x[i] - avg), 2);
            }

            return sum / (length - 1);
        }
        /// <summary>
        /// 均值方差
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Var(Vector x)
        {
            int length = x.Length;

            double avg = MathFExtension.Average(x);

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)(x[i] - avg), 2);
            }

            return sum / (length - 1);
        }

        /// <summary>
        /// 均值协方差
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="y">随机变量y</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double Cov(Vector x, Vector y)
        {
            int length = x.Length;

            if (length != y.Length) throw new Exception("结果数必须与结果概率数量一致");

            double xAvg = MathFExtension.Average(x);

            double yAvg = MathFExtension.Average(y);

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += (x[i] - xAvg) * (y[i] - yAvg);
            }

            return sum / (length - 1);
        }

        /// <summary>
        /// 均值协方差
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="y">随机变量y</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double Cov(double[] x, double[] y)
        {
            int length = x.Length;

            if (length != y.Length) throw new Exception("结果数必须与结果概率数量一致");

            double xAvg = MathFExtension.Average(x);

            double yAvg = MathFExtension.Average(y);

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += (x[i] - xAvg) * (y[i] - yAvg);
            }

            return sum / (length - 1);
        }

        /// <summary>
        /// 协方差
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="y">随机变量y</param>
        /// <param name="pd">概率分布概率</param>
        /// <param name="type"></param>
        /// <returns>协方差 >0 正相关； <0 负相关； =0不相关</returns>
        /// <exception cref="Exception"></exception>
        public static double Cov(double[] x, double[] y, double[] pd, RandomVariableType type = RandomVariableType.Discrete)
        {
            int length = x.Length;

            if (length != pd.Length || length != y.Length) throw new Exception("结果数必须与结果概率数量一致");

            double xExp = Exp(x, pd, type);

            double yExp = Exp(y, pd, type);

            double covValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                double val = (x[i] - xExp) * (y[i] - yExp);

                covValue += val * pd[i] * scale;
            }
            return covValue;
        }

        /// <summary>
        /// 协方差
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="y">随机变量y</param>
        /// <param name="pd">概率分布概率</param>
        /// <param name="type"></param>
        /// <returns>协方差 >0 正相关； <0 负相关； =0不相关</returns>
        /// <exception cref="Exception"></exception>
        public static double Cov(float[] x, float[] y, float[] pd, RandomVariableType type = RandomVariableType.Discrete)
        {
            int length = x.Length;

            if (length != pd.Length || length != y.Length) throw new Exception("结果数必须与结果概率数量一致");

            double xExp = Exp(x, pd, type);

            double yExp = Exp(y, pd, type);

            double covValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                double val = (x[i] - xExp) * (y[i] - yExp);

                covValue += val * pd[i] * scale;
            }
            return covValue;
        }

        /// <summary>
        /// 协方差
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="y">随机变量y</param>
        /// <param name="pd">概率分布概率</param>
        /// <param name="type"></param>
        /// <returns>协方差 >0 正相关； <0 负相关； =0不相关</returns>
        /// <exception cref="Exception"></exception>
        public static double Cov(Vector x, Vector y, Vector p, RandomVariableType type = RandomVariableType.Discrete)
        {
            int length = x.Length;

            if (length != p.Length || length != y.Length) throw new Exception("结果数必须与结果概率数量一致");

            double xExp = Exp(x, p, type);

            double yExp = Exp(y, p, type);

            double covValue = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                double val = (x[i] - xExp) * (y[i] - yExp);

                covValue += val * p[i] * scale;
            }
            return covValue;
        }

    }
}
