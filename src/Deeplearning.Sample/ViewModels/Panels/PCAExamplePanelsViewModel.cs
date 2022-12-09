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

        public static string Path = "./resources/testdata/iris_training.csv";       

        public Matrix source;// { get; set; }

        public PlotModel PlotModel { get; set; }

        public ScatterSeries Scatter { get; protected set; }
        public LineSeries Line { get; protected set; }

        public LineSeries Line2 { get; protected set; }

        public DelegateCommand LoadTestDataCommand { get; set; }
        public DelegateCommand PCACommand { get; set; }
        public DelegateCommand SVDPCACommand { get; set; }

     
        public PCAExamplePanelsViewModel()
        {
            PlotModel = new PlotModel();
     
            Scatter = new ScatterSeries() { MarkerType = MarkerType.Circle, MarkerFill = OxyColors.GreenYellow };

            Line = new LineSeries() {  LineStyle = LineStyle.Dot, MarkerSize = 8,MarkerType = MarkerType.Diamond, Color = OxyColors.Red };

            Line2 = new LineSeries() { LineStyle = LineStyle.Dot,MarkerSize=5,MarkerStroke= OxyColors.BurlyWood, MarkerType = MarkerType.Star, Color = OxyColors.BlueViolet };


            PlotModel.Series.Add(Scatter);

            PlotModel.Series.Add(Line);

            PlotModel.Series.Add(Line2);

            LoadTestDataCommand = new DelegateCommand(ExecuteLoadTestDataCommand);

            PCACommand = new DelegateCommand(ExecutePCACommand);

            SVDPCACommand = new DelegateCommand(ExecuteSVDPCACommand);
        }

     

        private void ExecuteReductionCommand()
        {
   
        }

        private void ExecutePCACommand()
        {
            int k = 1;

             var centralizedResult  = source.Standardized(0);
            //1.中心化
            Matrix centralizedMatrix = centralizedResult.matrix;

            //2.协方差矩阵
            Matrix covMatrix = centralizedMatrix.Cov();// Matrix.Dot(centralizedMatrix.T,centralizedMatrix) / (centralizedMatrix.Row)-1;// centralizedMatrix.Cov();

            //3.对协方差矩阵求特征值特征向量
            EigenDecompositionEventArgs result = Algebra.Eig(covMatrix);

            Matrix eigenVectors = result.eigenVectors;

            int dimension = eigenVectors.Row;

            //4.选取有效的特征值
            Matrix D = Matrix.Clip(eigenVectors, 0, 0, dimension, k);

            Matrix X = Matrix.Dot(centralizedMatrix, D);

            Matrix matrix = Matrix.Dot(X, D.T) + centralizedResult.means.T;         
        

            for (int i = 0; i < matrix.Row; i++)
            {
                Line.Points.Add(new DataPoint(matrix[i, 0], matrix[i, 1]));
            }
            PlotModel.InvalidatePlot(true);
        }

        private void ExecuteSVDPCACommand()
        {
            int k = 1;

            var standardResult = source.Standardized(0); //  centralInfo.matrix; //matrix;//

            Matrix standardMatrix = standardResult.matrix;
            //3.SVD
            SVDEventArgs result = Algebra.SVD(standardMatrix);

            Matrix eigenVectors = result.V;

            int dimension = eigenVectors.Row;

            //4.选取有效的特征值
            Matrix D = Matrix.Clip(eigenVectors, 0, 0, dimension, k);

            Matrix X = Matrix.Dot(standardMatrix, D);

            Matrix matrix = Matrix.Dot(X,D.T)+ standardResult.means.T;

            for (int i = 0; i < matrix.Row; i++)
            {
                Line2.Points.Add(new DataPoint(matrix[i,0],matrix[i,1]));
            }
            PlotModel.InvalidatePlot(true);
        }

        private void ExecuteLoadTestDataCommand()
        {

            Scatter.Points.Clear();
            Line.Points.Clear();
            Line2.Points.Clear();

            int dataCount = -1;

             source =  ReadLinearRegressionData(Path, dataCount);

            Scatter.Points?.Clear();

            for (int i = 0; i < source.Row; i++)
            {
                Scatter.Points.Add(new ScatterPoint(source[i,0], source[i,1]));
            }

            PlotModel.InvalidatePlot(true);
        }

        private Matrix ReadLinearRegressionData(string path, int count)
        {

            string[] lines = File.ReadAllLines(path);

            int dataCount = count > 0 ? count : lines.Length - 1;

            int dimension = 5;

            Random r = new Random();


            int startIndex = r.Next(0, lines.Length - dataCount);

            int endIndex = startIndex + dataCount;

            Matrix data = new Matrix(dataCount, dimension);

            for (int i = startIndex, j = 0; i < endIndex; i++, j++)
            {
                string content = lines[i + 1];
                string[] words = content.Split(',');
                // = 1;

                data[j, 0] = float.Parse(words[0]);
                data[j, 1] = float.Parse(words[1]);
                data[j, 2] = float.Parse(words[2]);
                data[j, 3] = float.Parse(words[3]);

                int tpye = 0;

                switch (words[4])
                {
                    case "setosa":
                        tpye = 0;
                        break;
                    case "versicolor":
                        tpye = 1;
                        break;
                    case "virginica":
                        tpye = 2;
                        break;
                    default:
                        break;
                }

                data[j, 4] = tpye;
            }
            return data;
        }
    }
}
