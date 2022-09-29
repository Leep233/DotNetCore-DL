using Deeplearning.Core.Math.Linear;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Interfaces
{
    public interface IEigenFilter
    {
        EigenDecompositionEventArgs Execute();
    }
}
