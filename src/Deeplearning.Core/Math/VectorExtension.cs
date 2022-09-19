using Deeplearning.Core.Extension;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public static class VectorExtension
    {
        public static bool IsOrthogonal(this Vector v1, Vector v2)
        {
            return ValueExtension.ZeroValidation(v1 * v2) == 0;
        }

        public static bool IsOrthogormal(this Vector v1, Vector v2)
        {

            bool result = true;

            if (IsOrthogonal(v1, v2))
            {
                if (!double.MinValue.Equals(MathF.Abs((float)(1 - v1.Norm(2)))))
                {
                    result = false;
                }
                else
                {
                    result = double.MinValue.Equals(MathF.Abs((float)(1 - v2.Norm(2))));
                }
            }
            else
            {
                result = false;
            }
            return result;
        }
    }
}
