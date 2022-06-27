using Deeplearning.Core.Math;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core
{
    public class PCA
    {
        private float[] source;
        private float[,] D;
        public float[] f(float[] x) {

            source = x;

            float [] c = new float[x.Length];
            return c;

        }

        public float[,] g(float[,] c) {
            return Matrix.Dot(D, c);
        }

        public float[] argmix(float[] x) {

            return x - g(c);
        }
    }
}
