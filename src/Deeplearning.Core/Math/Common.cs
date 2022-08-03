using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public class Common
    {
        public static float Sigmoid(float x) {

            return 1 / (1 + MathF.Exp(-x));
        
        }



    }
}
