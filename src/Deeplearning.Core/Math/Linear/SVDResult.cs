using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public class SVDResult
    {
        public Matrix U { get; set; }
        public Matrix S { get; set; }

        public Matrix V { get; set; }

        public SVDResult()
        {
            
        }

        public SVDResult(Matrix u, Matrix d , Matrix v)
        {
            this.U = u;
            this.S = d;
            this.V = v;
        }


        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("============[ SVD ]===========");
            sb.AppendLine("============Matrix.U===========");
            sb.AppendLine(U?.ToString());
            sb.AppendLine("============Matrix.D===========");
            sb.AppendLine(S?.ToString());
            sb.AppendLine("============Matrix.V===========");
            sb.AppendLine(V?.ToString());
   
            return sb.ToString();
        }

    }
}
