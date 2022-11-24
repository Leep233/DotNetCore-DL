using System;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public class EigenDecompositionEventArgs:DecomposeEventArgs
    {
        public Vector eigens;

        public Matrix eigenVectors;

        public EigenDecompositionEventArgs(Vector eigen, Matrix vectors)
        {
            this.eigens = eigen; 
            this.eigenVectors = vectors;

        }

        public EigenDecompositionEventArgs(Vector eigen, Vector[] vectors)
        {
            this.eigens = eigen;
            this.eigenVectors = new Matrix(vectors);
  
        }



        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(" [特征分解] ");
            sb.AppendLine("特征值（Eigen）");
            sb.AppendLine(eigens.ToString());
            sb.AppendLine("特征向量（Vectors）");
            sb.AppendLine(eigenVectors.ToString()); 

            return sb.ToString();
        }

        public override object Validate()
        {
            return this;
            //throw new NotImplementedException();
        }
    }
}
