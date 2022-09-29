using Deeplearning.Core.Example;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Core.Math.Models;
using OxyPlot;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace Deeplearning.Sample.ViewModels.Panels
{
    public class PCAExamplePanelsViewModel:BindableBase
    {

        public static string Path = "./resources/testdata/taxi-fare-train.csv";

        public Matrix source;// { get; set; }

        public PlotModel PlotModel { get; set; }

        public ScatterSeries Scatter { get; protected set; }

        public LineSeries Line { get; protected set; }

        public DelegateCommand LoadTestDataCommand { get; set; }
        public DelegateCommand PCACommand { get; set; }

        

        public PCAExamplePanelsViewModel()
        {
            PlotModel = new PlotModel();


            Scatter = new ScatterSeries() { MarkerType = MarkerType.Circle, MarkerFill = OxyColors.GreenYellow };

            Line = new LineSeries() { LineStyle = LineStyle.Solid, Color = OxyColors.Red };

            PlotModel.Series.Add(Scatter);

            PlotModel.Series.Add(Line);

            LoadTestDataCommand = new DelegateCommand(ExecuteLoadTestDataCommand);

            PCACommand = new DelegateCommand(ExecutePCACommand);
        }

        private void ExecutePCACommand()
        {
            PCA pca = new PCA();

            PCAEventArgs eventArgs = pca.Fit(source,1);


           // Debug.WriteLine(eventArgs);

          //  int r = .Column

            Matrix X = eventArgs.X;

           double rate = X[0,0] / X[0,1];

             

            Line.Points.Add(new DataPoint(0,0));
            Line.Points.Add(new DataPoint(rate * 50,50 ));
            // Line.Points.Add(p2);

            PlotModel.InvalidatePlot(true);
        }

        private void ExecuteLoadTestDataCommand()
        {
          source =  ReadLinearRegressionData(Path, 50);

            Scatter.Points?.Clear();

            for (int i = 0; i < source.Row; i++)
            {
                Scatter.Points.Add(new ScatterPoint(source[i,0], source[i, 1]));
            }
            PlotModel.InvalidatePlot(true);
        }

        private Matrix  ReadLinearRegressionData(string path, int count)
        {

            string[] lines = File.ReadAllLines(path);

            int dataCount = count > 0 ? count : lines.Length - 1;//count;
          //  Vector real = new Vector(dataCount);
           // int fc = lines[0].Split(',').Length - 2;
            Matrix data = new Matrix(dataCount, 2);

            for (int i = 0; i < dataCount; i++)
            {
                string content = lines[i + 1];
                string[] words = content.Split(',');
               // = 1;
               // data[i, 1] = float.Parse(words[1]);
               // data[i, 2] = float.Parse(words[2]);
               // data[i, 3] = float.Parse(words[3]);
                data[i, 0] = float.Parse(words[4]);//行程距离
                data[i, 1] = float.Parse(words[6]);//价格

            }
            return data;
        }

    }
}
