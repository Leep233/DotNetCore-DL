using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public class QREventArgs :DecomposeEventArgs
    {
        public Matrix Q { get; set; }

        public Matrix R { get;  set; }

        public QREventArgs ()
        {

        }

        public QREventArgs (Matrix q,Matrix r)
        {
            this.Q = q;
            this.R = r; 
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("============Matrix.Q===========");
            sb.AppendLine(this.Q.ToString());
            sb.AppendLine("============Matrix.R===========");
            sb.AppendLine(this.R.ToString());
            return sb.ToString();
        }

        public override object Validate()
        {
            return Q * R;
        }
    }
}
