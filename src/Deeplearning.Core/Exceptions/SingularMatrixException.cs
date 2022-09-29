using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Exceptions
{
    public class SingularMatrixException:Exception
    {
        public SingularMatrixException():base("奇异矩阵")
        {

        }

        public SingularMatrixException(string message):base(message)
        {

        }
    }
}
