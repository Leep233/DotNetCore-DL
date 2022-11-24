using Deeplearning.Core.Math.Common;
using Deeplearning.Core.Math.Probability;
using System;

namespace Deeplearning.Core.Math
{
    /// <summary>
    /// 信息论相关函数
    /// </summary>
    public class InformationTheory
    {

        /// <summary>
        /// 信息量
        /// </summary>
        /// <param name="p">概率</param>
        /// <param name="unit"></param>
        /// <returns></returns>
        public static double Information(double p, InformationUnit unit = InformationUnit.Nats)
        {
            return -CommonFunctions.MathfLog(unit)((float)p);
        }

        /// <summary>
        /// 信息熵
        /// </summary>
        /// <param name="x">参数</param>
        /// <param name="p">参数对应概率</param>
        /// <param name="unit"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        public static double InformationEntropy(double[] x, double[] p, InformationUnit unit = InformationUnit.Nats, RandomVariableType type = RandomVariableType.Discrete)
        {
            double expValue = 0;

            int length = x.Length;

            Func<float, float> logFun = CommonFunctions.MathfLog(unit);

            double scale = CommonFunctions.ProbabilityDistributionScale(type);

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
        public static double InformationEntropy(Vector x, Vector p, InformationUnit unit = InformationUnit.Nats, RandomVariableType mode = RandomVariableType.Discrete)
        {
            double expValue = 0;

            int length = x.Length;

            double scale = CommonFunctions.ProbabilityDistributionScale(mode);

            Func<float, float> logFun = CommonFunctions.MathfLog(unit);

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
        /// <typeparam name="T"></typeparam>
        /// <param name="pdf"></param>
        /// <param name="x"></param>
        /// <param name="unit"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static double InformationEntropy<T>(Func<T, T> pdf, T x, InformationUnit unit = InformationUnit.Nats, RandomVariableType mode = RandomVariableType.Discrete)
        {
            T p = pdf(x);

            double result = 0;

            if (x is double[] x_d_values && p is double[] pd_d_values)
            {
                result = InformationEntropy(x_d_values, pd_d_values,unit,mode);
            }
            else if (x is double[] x_f_values && p is double[] pd_f_values)
            {
                result = InformationEntropy(x_f_values, pd_f_values, unit, mode);
            }
            else if (x is double[] x_v_values && p is double[] pd_v_values)
            {
                result = InformationEntropy(x_v_values, pd_v_values, unit, mode);
            }
            else {
                throw new ArgumentException("this arguments type is not supported!!");
            }
            return result;
        }


        /// <summary>
        /// 信息熵
        /// </summary>
        /// <param name="x"></param>
        /// <param name="p"></param>
        /// <param name="unit"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static double InformationEntropy(Func<Vector, Vector> probabilityDistributionFunction, Vector x, InformationUnit unit = InformationUnit.Nats, RandomVariableType mode = RandomVariableType.Discrete)
        {
            Vector p = probabilityDistributionFunction(x);

            return InformationEntropy(x, p, unit, mode);
        }


        /// <summary>
        ///  kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="pd1">概率分布1</param>
        /// <param name="pd2">概率分布1</param>
        /// <param name="unit">信息值单位默认：nats</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns></returns>
        public static double KLDivergence(float[] x, float[] pd1, float[] pd2, InformationUnit unit = InformationUnit.Nats, RandomVariableType type = RandomVariableType.Discrete)
        {
            int length = x.Length;

            double result = 0;

            var logFunction = CommonFunctions.MathfLog(unit);

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                double temp = logFunction(pd1[i]) - logFunction(pd2[i]);

                result += x[i] * scale * temp;
            }

            return result;
        }

        /// <summary>
        ///  kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="pd1">概率分布1</param>
        /// <param name="pd2">概率分布1</param>
        /// <param name="unit">信息值单位默认：nats</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns></returns>
        public static double KLDivergence(double[] x, double[] pd1, double[] pd2, InformationUnit unit = InformationUnit.Nats, RandomVariableType type = RandomVariableType.Discrete)
        {
            int length = x.Length;

            double result = 0;

            var logFunction = CommonFunctions.MathfLog(unit);

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                double temp = logFunction((float)pd1[i]) - logFunction((float)pd2[i]);

                result += x[i] * scale * temp;
            }

            return result;
        }

        /// <summary>
        ///  kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="pd1">概率分布1</param>
        /// <param name="pd2">概率分布1</param>
        /// <param name="unit">信息值单位默认：nats</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns></returns>
        public static double KLDivergence(Vector x, Vector pd1, Vector pd2, InformationUnit unit = InformationUnit.Nats, RandomVariableType type = RandomVariableType.Discrete)
        {
            int length = x.Length;

            double result = 0;

            var logFunction = CommonFunctions.MathfLog(unit);

            float scale = CommonFunctions.ProbabilityDistributionScale(type);

            for (int i = 0; i < length; i++)
            {
                double temp = logFunction((float)pd1[i]) - logFunction((float)pd2[i]);

                result += x[i] * temp * scale;
            }

            return result;
        }

        /// <summary>
        /// kl散度。同一个随机变量有两个独立的概率分布，用来衡量两个分布的差异
        /// </summary>
        /// <param name="x">随机变量X</param>
        /// <param name="pdf1">概率分布函数1</param>
        /// <param name="pdf2">概率分布函数2</param>
        /// <param name="unit">信息值单位默认：nats</param>
        /// <param name="type">随机变量类型：离散型/连续型；默认离散型</param>
        /// <returns>kl散度</returns>
        public static double KLDivergence<T>(T x, Func<T, T> pdf1, Func<T, T> pdf2, InformationUnit unit = InformationUnit.Nats, RandomVariableType type = RandomVariableType.Discrete)
        {
            T pd1 = pdf1(x);

            T pd2 = pdf2(x);

            double result = 0;

            if (x is double[] x_D_Value && pd1 is double[] pd1_D_Value && pd2 is double[] pd2_D_Value)
            {
                result = KLDivergence(x_D_Value, pd1_D_Value, pd2_D_Value, unit, type);
            }
            else if (x is float[] x_F_Value && pd1 is float[] pd1_F_Value && pd2 is float[] pd2_F_Value)
            {
                result = KLDivergence(x_F_Value, pd1_F_Value, pd2_F_Value, unit, type);
            }
            else if (x is Vector x_V_Value && pd1 is Vector pd1_V_Value && pd2 is Vector pd2_V_Value)
            {
                result = KLDivergence(x_V_Value, pd1_V_Value, pd2_V_Value, unit, type);
            }
            else
            {
                throw new ArgumentException("This parameter is not supported!!(support:float[] double[],Vector)");
            }
            return result;
        }


        /// <summary>
        ///  交叉熵
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static double CrossEntropy(double[] x, double[] p1, double[] p2, InformationUnit unit = InformationUnit.Nats, RandomVariableType mode = RandomVariableType.Discrete)
        {
            int length = x.Length;

            double result = 0;

            double entropy = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(mode);

            Func<float, float> logFunction = CommonFunctions.MathfLog(unit);

            for (int i = 0; i < length; i++)
            {
                double p1LogValue = logFunction((float)p1[i]);

                double info = -p1LogValue;

                double temp = p1LogValue - logFunction((float)p2[i]);

                double scaleValue = x[i] * scale;

                result = temp * scaleValue;

                entropy += info * scaleValue;
            }

            return entropy + result;
        }

        /// <summary>
        /// 交叉熵
        /// </summary>
        /// <param name="x">随机变量x</param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static double CrossEntropy(Vector x, Vector p1, Vector p2, InformationUnit unit = InformationUnit.Nats, RandomVariableType mode = RandomVariableType.Discrete)
        {
            int length = x.Length;

            double result = 0;

            double entropy = 0;

            float scale = CommonFunctions.ProbabilityDistributionScale(mode);

            Func<float, float> logFunction = CommonFunctions.MathfLog(unit);

            for (int i = 0; i < length; i++)
            {

                double p1LogValue = logFunction((float)p1[i]);
                //P1分布对应的信息量
                double infoAmount = -p1LogValue;

                double temp = p1LogValue - logFunction((float)p2[i]);

                double scaleValue = x[i] * scale;

                result = scaleValue * temp;

                entropy += scaleValue * infoAmount;
            }

            return entropy + result;
        }

        public static double MES(Vector x, Vector y)
        {
            int length = x.Length;

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)(x[i] - y[i]), 2);
            }

            return MathF.Sqrt((float)sum) / length;
        }

        public static double MES(float[] x, float[] y)
        {
            int length = x.Length;

            double sum = 0;

            for (int i = 0; i < length; i++)
            {      
                sum += MathF.Pow((float)(x[i] - y[i]),2);
            }

            return MathF.Sqrt((float)sum) / length;
        }

        public static double MES(double[] x, double[] y)
        {
            int length = x.Length;

            double sum = 0;

            for (int i = 0; i < length; i++)
            {
                sum += MathF.Pow((float)(x[i] - y[i]), 2);
            }

            return MathF.Sqrt((float)sum) / length;
        }

    }
}
