using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Wpf;
using Prism.Commands;
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

        public DelegateCommand FlipCoinCommand { get; set; }

        public PlotModel PlotModel { get; set; }

        public TheoryProbabilityPanelViewModel()
        {
            PlotModel = new PlotModel();


            FlipCoinCommand = new DelegateCommand(ExecuteFlipCoinCommand);

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

        private void ExecuteFlipCoinCommand()
        {
            PlotModel.Series.Clear();

            int testNum = 1000;

            int flipCount = 50;

            int [] points = new int[flipCount+1]; 

            for (int i = 0; i < testNum; i++)
            {
                points[FlipCoin(flipCount)]++;
            }

            LinearBarSeries linearBarSeries = new LinearBarSeries();

            linearBarSeries.BarWidth = 15;

            for (int i = 0; i < points.Length; i++)
            {
                linearBarSeries.Points.Add(new DataPoint(i, points[i])) ;
            }

            PlotModel.Series.Add(linearBarSeries);

            PlotModel.InvalidatePlot(true);
        }

        /// <summary>
        /// 抛硬币
        /// </summary>
        /// <param name="count">抛掷次数</param>
        /// <returns></returns>
        private int FlipCoin(int count=10,float sigma=0.5f)
        {
            int sum = 0;

            Random random = new Random();          

            for (int i = 0; i < count; i++)
            {
                //0表示正面
                if (random.NextDouble() <= sigma) {sum++;}
            }
            return sum;
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
