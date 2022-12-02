using Deeplearning.Core.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public abstract class DecomposeEventArgs:EventArgs, IVerifiable
    {
        public virtual object Validate() { return ""; }
    }
}
