using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.LinearAlgebra
{
    public class SVDResult
    {
        public Matrix U { get; set; }
        public Matrix D { get; set; }

        public Matrix V { get; set; }

        public SVDResult()
        {
            
        }

        public SVDResult(Matrix u, Matrix d , Matrix v)
        {
            this.U = u;
            this.D = d;
            this.V = v;
        }


        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============Matrix.U===========");
            sb.AppendLine(U?.ToString());
            sb.AppendLine("============Matrix.D===========");
            sb.AppendLine(D?.ToString());

            sb.AppendLine("============Matrix.V===========");
            sb.AppendLine(V?.ToString());

            return sb.ToString();
        }

    }
}
