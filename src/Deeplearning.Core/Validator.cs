using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core
{
    internal class Validator
    {
        public const double MIN_VALUE = 1E-15;
        public static double ZeroValidation(double value)
        {
            return MathF.Abs((float)value) <= MIN_VALUE ? 0 : value;
        }
    }
}
