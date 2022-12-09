using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Sample.ViewModels
{
    public abstract class DistributionBase:BindableBase
    {
        public DelegateCommand DistributionCommand { get; set; }


        public DistributionBase()
        {
            DistributionCommand = new DelegateCommand(ExecuteDistributionCommand);
        }

        protected virtual void ExecuteDistributionCommand()
        {
           
        }
    }
}
