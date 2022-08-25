using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Attributes
{
    public class CompletionAttribute:Attribute
    {
        public readonly bool completed;

        public CompletionAttribute(bool isCompleted = true)
        {
            completed = isCompleted;
        }

    }
}
