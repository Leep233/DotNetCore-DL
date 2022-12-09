using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Wpf;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Sample.ViewModels.Panels
{
    public class TheoryProbabilityPanelViewModel:BindableBase
    {
        public NormalDistributionViewModel NormalDistribution { get; set; }

        public LaplaceViewModel Laplace { get; set; }

        public EmpiricalViewModel Empirical { get; set; }

        public CommonFunctionListViewModel CommonFunction { get; set; }


        public PlotModel PlotModel { get; set; }



        public TheoryProbabilityPanelViewModel()
        {
            PlotModel = new PlotModel();

            PlotModel.TitleHorizontalAlignment = TitleHorizontalAlignment.CenteredWithinPlotArea;

            Empirical = new EmpiricalViewModel();
           
            Empirical.OnSeriesChanged += OnSeriesChanged;

            CommonFunction = new CommonFunctionListViewModel();
            CommonFunction.OnSeriesChanged += OnSeriesChanged;

            CommonFunction.OnSelectedFunctionChanged += (f => {
                Empirical.DeltaFunction = new Func<float, float>(f); 
            });

            Laplace = new LaplaceViewModel();
            Laplace.OnSeriesChanged += OnSeriesChanged;


         


            NormalDistribution = new NormalDistributionViewModel();
            NormalDistribution.OnSeriesChanged += OnSeriesChanged;
           
        }

        private void OnSeriesChanged(FunctionSeries old, FunctionSeries @new)
        {
            if(old !=null)
                PlotModel.Series.Remove(old);

            PlotModel.Series.Add(@new);

            PlotModel.InvalidatePlot(true);
        }
    }
}
