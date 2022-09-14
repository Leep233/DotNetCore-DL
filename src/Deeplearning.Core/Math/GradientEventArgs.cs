using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public struct GradientEventArgs
    {
        public float x;
        public float y;
        public float k;

        public override string ToString()
        {
            return $"({x.ToString("F4")},{y.ToString("F4")}) 斜率/导数：{k.ToString("F4")}";
        }
    }
}
