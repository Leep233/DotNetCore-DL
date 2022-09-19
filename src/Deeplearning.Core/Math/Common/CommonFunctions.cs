using Deeplearning.Core.Math.Probability;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Common
{
    public static class CommonFunctions
    {
        public static float Sigmoid(float x)
        {
            return 1 / (1 + MathF.Exp(-x));
        }

        public static float Softplus(float x) { 
         return MathF.Log(1+ MathF.Exp(x));
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
    }
}
