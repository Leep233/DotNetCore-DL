using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Common
{
    public class CommonFunctions
    {
        public static float Sigmoid(float x)
        {
            return 1 / (1 + MathF.Exp(-x));
        }

        public static float Softplus(float x) { 
         return MathF.Log(1+ MathF.Exp(x));
        }


    }
}
