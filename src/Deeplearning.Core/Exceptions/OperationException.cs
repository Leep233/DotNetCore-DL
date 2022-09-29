using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Exceptions
{
    public class OperationException:Exception
    {
        public OperationException():base("不支持的操作")
        {
            
        }
        public OperationException(string message):base(message)
        {

        }
        
    }
}
