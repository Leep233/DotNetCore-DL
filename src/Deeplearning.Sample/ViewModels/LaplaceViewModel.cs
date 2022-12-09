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
    public sealed class LaplaceViewModel: DistributionBase
    {
        private float u = 0.5f;

        public float U
        {
            get { return u; }
            set { u = value; RaisePropertyChanged("U"); }
        }

        private float y = 0.5f;

        public float Y
        {
            get { return u; }
            set { u = value; RaisePropertyChanged("Y"); }
        }

        private FunctionSeries series;

        public event Action<FunctionSeries, FunctionSeries> OnSeriesChanged;

        protected override void ExecuteDistributionCommand()
        {
            double dx = 0.01;

            FunctionSeries newSeries = new FunctionSeries(x => ProbabilityDistribution.Laplace((float)x, U, Y),  - 2,  + 2, dx)
            {
                Color = OxyColors.Lavender,
                Title = $"正态分布(u={U.ToString("F2")},a={Y.ToString("F2")})"
            };

            OnSeriesChanged?.Invoke(series, newSeries);

            series = newSeries;
        }

    }
}
