using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Sample.ViewModels.Components
{
    public class IOComponentViewModel:BindableBase
    {
        private string inputContent;

        public string InputContent
        {
            get { return inputContent; }
            set { inputContent = value;RaisePropertyChanged("InputContent"); }
        }
        private string outputContent;

        public string OutputContent
        {
            get { return outputContent; }
            set { outputContent = value; RaisePropertyChanged("OutputContent"); }
        }
   
    
    
    }
}
