using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public struct GradientInfo
    {
        public float x;
        public float y;
        public float k;

        public override string ToString()
        {
            return $"({x.ToString("F4")},{y.ToString("F4")}) 斜率/导数：{k.ToString("F4")}";
        }
    }

    public struct Gradient3DInfo
    {
        public float x;
        public float y;
        public float z;
        internal Vector grad;

        public override string ToString()
        {
            return $"({x.ToString("F4")},{y.ToString("F4")},{z.ToString("F4")})";
        }

    }
}
