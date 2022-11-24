using System;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public class PCAEventArgs : DecomposeEventArgs
    {
        public Matrix D { get; set; }

        public Matrix X { get; set; }


        public PCAEventArgs(Matrix x,Matrix d)
        {
            D = d;
            X = x;
        }
        public override object Validate()
        {
            throw new NotImplementedException();
        }

        public override string ToString()
        {
           StringBuilder sb = new StringBuilder();
            sb.AppendLine("========= X =========");
            sb.AppendLine(X.ToString());
            sb.AppendLine("========= D =========");
            sb.AppendLine(D.ToString());
            return sb.ToString();
        }
    }
}
