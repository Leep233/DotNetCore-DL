using Deeplearning.Core.Math;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core
{
    public class PCA
    {
        public double[,] Source { get; set; }

        

        public float[] ArgMin(float[] vector) {
           throw  new NotFiniteNumberException();


        }

        public float[,] Decompression(float[] c) 
        {
            float[,] D = BuidleDMatrix(c);

            return Linear.Dot(D, c);
        }

        private float[,] BuidleDMatrix(float[] c) 
        {

            float[,] D = new float[Source.GetLength(1), c.Length];

            return D;
        }


        public float[] Compression() { 
            
            int count = Source.GetLength(0);

            float[] result = new float[count];

            return result;
        }
    }
}
