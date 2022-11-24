using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public class SVDEventArgs:DecomposeEventArgs
    {
        public Matrix U { get; set; }
        public Matrix S { get; set; }
        public Matrix V { get; set; }

        public SVDEventArgs()
        {
            
        }

        public SVDEventArgs(Matrix u, Matrix d , Matrix v)
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
            sb.AppendLine(U.ToString());
            sb.AppendLine("============Matrix.S===========");
            sb.AppendLine(S.ToString());
            sb.AppendLine("============Matrix.V===========");
            sb.AppendLine(V.ToString());   
            return sb.ToString();
        }

        public override object Validate()
        {
           // Matrix matrix = Matrix.Inv();

            return U * S * V.T;
        }
    }
}
