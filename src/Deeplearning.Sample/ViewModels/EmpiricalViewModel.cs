using OxyPlot;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Sample.ViewModels
{
    public class EmpiricalViewModel: DistributionBase
    {

        private float u;

        public float U
        {
            get { return u; }
            set { u = value; RaisePropertyChanged("U"); }
        }


        public  Func<float, float> DeltaFunction;

        private FunctionSeries series;

        public event Action<FunctionSeries, FunctionSeries> OnSeriesChanged;

        protected override void ExecuteDistributionCommand()
        {
            double dx = 0.01;

            FunctionSeries newSeries = new FunctionSeries(x => DeltaFunction?.Invoke((float)(x - U))??0, -2, +2, dx)
            {
                Color = OxyColors.Gold,
            };

            OnSeriesChanged?.Invoke(series, newSeries);

            series = newSeries;
        }
    }
}
