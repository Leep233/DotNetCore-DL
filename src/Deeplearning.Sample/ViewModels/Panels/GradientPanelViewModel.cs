using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Sample.Utils;
using OxyPlot;
using OxyPlot.Annotations;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Sample.ViewModels.Panels
{
    public class GradientPanelViewModel : BindableBase
    {
        private string message;

        public string Message
        {
            get { return message; }
            set { message = value; RaisePropertyChanged("Message"); }
        }


        //0.5x^2 + 2x -3
        public Func<double, double> linearEquation = new Func<double, double>(x => (0.5 * x * x + 2 * x - 3));

        public PlotModel PlotModel { get; set; }

        public LineSeries lineSeries;

        public FunctionSeries functionSeries;

        public ScatterSeries scatterSeries;

        public DelegateCommand<string> GradientDescentCommand { get; set; }

        public GradientPanelViewModel()
        {
            PlotModel = new PlotModel();

     

            GradientDescentCommand = new DelegateCommand<string>(ExecuteGradientDescentCommand);
        }

        private void ExecuteGradientDescentCommand(string arg)
        {
            if (int.TryParse(arg, out int type))
            {
                switch (type)
                {
                    case 0:
                        GradientDescent2D();
                        break;
                    default:
                        GradientDescentMultiD();
                        break;
                }
            }


        }



        private void GradientDescentMultiD()
        {

        }

        private async void GradientDescent2D()
        {
            //绘制线性方程
            DrawEquation();

            Vector vector = await Gradient.GradientDescent(linearEquation,3, GradientParams.Default, OnGradientChangedCallback);
        }



        private async Task OnGradientChangedCallback(GradientEventArgs eventArgs)
        {
            await Task.Delay(30);

            var points = OxyPlotHelper.GetTangentLinePoints(eventArgs, 3);

            lineSeries.Points.Clear();

            scatterSeries.Points.Clear();    

            lineSeries.Points.Add(points.p1);

            lineSeries.Points.Add(points.p2);

            scatterSeries.Points.Add(new ScatterPoint(eventArgs.x, eventArgs.y));      

            PlotModel.InvalidatePlot(true);

            Message = eventArgs.ToString();
        }


      
        private void DrawEquation()
        {

            lineSeries = new LineSeries() { Color = OxyColors.Red, LineStyle = LineStyle.Dash };

            scatterSeries = new ScatterSeries() { MarkerType = MarkerType.Circle };

            PlotModel.Series.Add(lineSeries);

            PlotModel.Series.Add(scatterSeries);

            if (functionSeries != null)
                PlotModel.Series.Remove(functionSeries);

            functionSeries = new FunctionSeries(linearEquation, -8, 4, 0.1, "0.5x^2 + 2x -3")
            {
                Color = OxyColors.BlueViolet,
                MarkerType = MarkerType.None,
                LineStyle = LineStyle.Solid,
            };

            PlotModel.Series.Add(functionSeries);

            PlotModel.InvalidatePlot(true);
        }
    }
}
