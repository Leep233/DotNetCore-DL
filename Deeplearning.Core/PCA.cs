using Deeplearning.Core.Math;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core
{
    public class PCA
    {
        public double[,] Source { get; set; }

        

        public double[] ArgMin(double [] vector) {
           throw  new NotFiniteNumberException();


        }

        public double[,] Decompression(double [] c) 
        {
            double[,] D = BuidleDMatrix(c);

            return Linear.Dot(D, c);
        }

        private double[,] BuidleDMatrix(double [] c) 
        {

            double[,] D = new double[Source.GetLength(1), c.Length];

            return D;
        }


        public double[] Compression() { 
            
            int count = Source.GetLength(0);

            double [] result = new double[count];

            return result;
        }
    }
}
