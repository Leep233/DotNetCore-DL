using Deeplearning.Core.Example;
using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
using OxyPlot;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Diagnostics;
using System.IO;

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

        public DelegateCommand ReductionCommand { get; set; }

        public PCAExamplePanelsViewModel()
        {
            PlotModel = new PlotModel();
     
            Scatter = new ScatterSeries() { MarkerType = MarkerType.Circle, MarkerFill = OxyColors.GreenYellow };

            Line = new LineSeries() {  LineStyle = LineStyle.Dot,MarkerType = MarkerType.Diamond, Color = OxyColors.Red };

            PlotModel.Series.Add(Scatter);

            PlotModel.Series.Add(Line);

            LoadTestDataCommand = new DelegateCommand(ExecuteLoadTestDataCommand);

            PCACommand = new DelegateCommand(ExecutePCACommand);

            ReductionCommand = new DelegateCommand(ExecuteReductionCommand);
        }

        private PCAEventArgs pcaEventArgs = null;

        private void ExecuteReductionCommand()
        {
            if (pcaEventArgs is null) return;


            Matrix matrix = pcaEventArgs.D * pcaEventArgs.X;

            for (int i = 0; i < matrix.Row; i++)
            {

                Line.Points.Add(new DataPoint(matrix[i, 0], matrix[i, 1]));

               // Scatter2.Points.Add(new ScatterPoint(matrix[i, 0], matrix[i, 1]));
            }
            PlotModel.InvalidatePlot(true);
        }

        private void ExecutePCACommand()
        {
            PCA pca = new PCA();

         pcaEventArgs = pca.SVDFit(source,1);

            //     pcaEventArgs = pca.EigFit(source, 1);

            ReductionCommand.Execute();
        }

        private void ExecuteLoadTestDataCommand()
        {

            Scatter.Points.Clear();
            Line.Points.Clear();

            int dataCount = 20;

             source =  ReadLinearRegressionData(Path, dataCount);

            Scatter.Points?.Clear();

            for (int i = 0; i < source.Row; i++)
            {
                Scatter.Points.Add(new ScatterPoint(source[i,0], source[i,1]));
            }

            PlotModel.InvalidatePlot(true);
        }

        private Matrix  ReadLinearRegressionData(string path, int count)
        {

            string[] lines = File.ReadAllLines(path);

            int dataCount = count > 0 ? count : lines.Length - 1;

            Random r = new Random();
            

            int startIndex = r.Next(0, lines.Length - dataCount);

            int endIndex = startIndex + dataCount;

            Matrix data = new Matrix(dataCount,2);

            for (int i = startIndex,j=0 ; i < endIndex; i++,j++)
            {
                string content = lines[i + 1];
                string[] words = content.Split(',');
                // = 1;

                data[j,0] = float.Parse(words[4]);//行程距离
                data[j,1] = float.Parse(words[3]);//价格

               // data[ 0,i] = float.Parse(words[1]);
               // data[ 1,i] = float.Parse(words[2]);
               // data[ 2,i] = float.Parse(words[3]);
               // data[ 3,i] = float.Parse(words[4]);//行程距离
               // data[ 4,i] = float.Parse(words[6]);//价格

            }
            return data;
        }

    }
}
