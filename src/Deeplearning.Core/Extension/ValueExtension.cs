using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Extension
{
    public static class ValueExtension
    {
        public const double MIN_VALUE = 1E-15;
        public static double ZeroValidation(double value)
        {
            return MathF.Abs((float)value) <= MIN_VALUE ? 0 : value;
        }

        public static float ZeroValidation(float value)
        {
            return MathF.Abs(value) <= MIN_VALUE ? 0 : value;
        }

        public static bool ValueValidate<T>(T value)
        {
            return (value is double[] || value is float[] || value is Vector);
        }
        public static bool ValueValidate<T>(this object sender, T value)
        {
            return (value is double[] || value is float[] || value is Vector);
        }
    }
}
