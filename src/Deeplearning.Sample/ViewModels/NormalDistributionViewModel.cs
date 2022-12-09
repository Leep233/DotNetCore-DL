using Deeplearning.Core.Math;
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
    public sealed class NormalDistributionViewModel: DistributionBase
    {

        private float u = 0.5f;

        public float U
        {
            get { return u; }
            set { u = value;RaisePropertyChanged("U"); }
        }

        private float a = 0.5f;

        public float A
        {
            get { return u; }
            set { u = value; RaisePropertyChanged("A"); }
        }

        private FunctionSeries series;

        public event Action<FunctionSeries, FunctionSeries> OnSeriesChanged;

   

        protected override void ExecuteDistributionCommand()
        {
            double dx = 0.01;

            FunctionSeries newSeries = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, U, A), A - 2, A + 2, dx)
            {
                Color = OxyColors.Orange,
                Title = $"正态分布(u={U.ToString("F2")},a={A.ToString("F2")})"
            };

            OnSeriesChanged?.Invoke(series, newSeries);

            series = newSeries;

        }
    }
}
