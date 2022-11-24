using System;

namespace Deeplearning.Core.Math.Linear
{
    public class EigenEventArgs: DecomposeEventArgs
    {
        public double eigen { get;private set; }

        public Vector vector { get;private set; }

        public EigenEventArgs(double eigen,Vector vector)
        {
            this.eigen = eigen;
            this.vector = vector;    
        }

        public override string ToString()
        {
            return $"Eigen:{eigen.ToString("F4")} Eigen Vector : {vector}";
        }

        public override object Validate()
        {
            throw new NotImplementedException();
        }
    }
}
