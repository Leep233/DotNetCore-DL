using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.LinearAlgebra
{
    public class EigResult:DecomposeResult
    {
        public Matrix Eigen { get; set; }

        public Matrix Vectors { get; set; }

        public EigResult()
        {

        }

        public EigResult(Matrix eigen, Matrix vectors)
        {
            this.Eigen = eigen; 
            this.Vectors = vectors;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============Eigen Value===========");
            sb.AppendLine(Eigen.ToString());
            sb.AppendLine("============Eigen Vectors===========");
            sb.AppendLine(Vectors.ToString());

            return sb.ToString();
        }
    }
}
