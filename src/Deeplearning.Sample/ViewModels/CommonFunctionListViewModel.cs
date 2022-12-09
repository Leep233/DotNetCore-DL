using Deeplearning.Core.Math.Common;
using Deeplearning.Core.Math.Probability;
using OxyPlot;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Sample.ViewModels
{
    public class CommonFunctionListViewModel:BindableBase
    {

        private int selectedIndex;

        public int SelectedIndex
        {
            get { return selectedIndex; }
            set { selectedIndex = value;RaisePropertyChanged("SelectedIndex"); SelectedFunction(selectedIndex); }
        }

        public DelegateCommand FunctionCommand { get; set; }

        public Func<float, float> SelcetedFunction { get; private set; }


        private FunctionSeries series;


        public event Action<Func<float, float>> OnSelectedFunctionChanged;


        public event Action<FunctionSeries, FunctionSeries> OnSeriesChanged;
        public CommonFunctionListViewModel()
        {
            FunctionCommand = new DelegateCommand(ExecuteFunctionCommand);

            SelectedFunction(0);
        }

        private void SelectedFunction(int index) {

            switch ((CommonFunction)index)
            {
           
                case CommonFunction.Softplus:
                    SelcetedFunction = new Func<float, float>(CommonFunctions.Softplus);
                    break;
                case CommonFunction.Sigmoid:
                default:
                    SelcetedFunction = new Func<float, float>(CommonFunctions.Sigmoid);
                    break;
            }
            OnSelectedFunctionChanged?.Invoke(SelcetedFunction);
        }

        private void ExecuteFunctionCommand()
        {
            double dx = 0.01;

            FunctionSeries newSeries = new FunctionSeries(x => SelcetedFunction?.Invoke((float)x)??0,  -5,  + 5, dx)
            {
                Color = OxyColors.PaleGoldenrod,          
            };

            OnSeriesChanged?.Invoke(series, newSeries);

            series = newSeries;
        }

        internal enum CommonFunction { 
        
            Sigmoid,
            Softplus
        
        }

        



    }
}
