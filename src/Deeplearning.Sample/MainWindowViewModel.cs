using Deeplearning.Core.Math;
using Deeplearning.Sample.Utils;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Sample
{
    public class MainWindowViewModel : BindableBase
    {

        public const int PointSize = 3;

   
        public const int MajorStep = 1;

        private float[,] SampleMatrix;


        private string message;
        public string Message
        {
            get { return message; }
            set { message = value; RaisePropertyChanged("Message"); }
        }

        private string sourceMatrix;
        public string SourceMatrix
        {
            get { return sourceMatrix; }
            set { sourceMatrix = value; RaisePropertyChanged("SourceMatrix"); }
        }

        public OxyPlotView LeftPlotView { get; set; }
        public OxyPlotView RightPlotView { get; set; }

        public DelegateCommand UpdateSourceMatrixCommannd { get; set; }

        public DelegateCommand ComputeCommand { get; set; }

        public DelegateCommand GradientCommand { get; set; }

        public DelegateCommand Gradient3DCommand { get; set; }

        public DelegateCommand NormalDistriutionCommand { get; set; }



        public MainWindowViewModel()
        {


            LeftPlotView = new OxyPlotView(OxyColors.Orange,OxyColors.DeepPink);

            RightPlotView = new OxyPlotView(OxyColors.GreenYellow, OxyColors.DodgerBlue);


            ExecuteUpdateSourceMatrixCommannd();

            UpdateSourceMatrixCommannd = new DelegateCommand(ExecuteUpdateSourceMatrixCommannd);

            ComputeCommand = new DelegateCommand(ExecuteComputeCommand);

            GradientCommand = new DelegateCommand(ExecuteGradientCommand);

            Gradient3DCommand = new DelegateCommand(ExecuteGradient3DCommand);

            NormalDistriutionCommand = new DelegateCommand(ExecuteNormalDistriutionCommand);
        }

        private void ExecuteNormalDistriutionCommand()
        {
            FunctionSeries series1 = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, 0.5f, 0.5f), -3, 3, 0.1) { 
            Color = OxyColors.Red,
                Title = "正态分布(u=0.5,a=0.5)",
             };

            FunctionSeries series2 = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, 1, 0.5f), -3, 3, 0.1)
            {
                Color = OxyColors.Orange,
                Title = "正态分布(u=1,a=0.5f)"
            };

            FunctionSeries series3 = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, 0.5f, 1f), -3, 3, 0.1)
            {
                Color = OxyColors.DeepSkyBlue,
                Title = "标准正态分布(u=0.5,a=1f)"
            };

            FunctionSeries series4 = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, 0, 1), -3, 3, 0.1) {
                Color = OxyColors.Green, Title = "标准正态分布(u=0,a=1)"
            };

            LeftPlotView.AddSeries(series1);
            LeftPlotView.AddSeries(series2);
            LeftPlotView.AddSeries(series3);
            LeftPlotView.AddSeries(series4);

            LeftPlotView.UpdateView();
        }

        List<Gradient3DInfo> v3Points = new List<Gradient3DInfo>();        
        

        private async void ExecuteGradient3DCommand()
        {
            //v3Points.Clear();

            Func<Vector2D, float> original = new Func<Vector2D, float>(vector => MathF.Pow(vector.x, 2) + MathF.Pow(vector.y, 2));

            await Linear.GradientDescentTaskAsync(500, original,OnGradient3DChangedCallback);

            using (StreamWriter writer = File.CreateText("point.txt")) {

                

                for (int i = 0; i < v3Points.Count; i++)
                {
                    writer.WriteLine(v3Points[i].ToString());
                }

            }

            Message = "done...";

        }

        private void OnGradient3DChangedCallback(Gradient3DInfo value)
        {
            Message = value.ToString();
            v3Points.Add(value);
        }

        ~MainWindowViewModel() {
           
        }

        /// <summary>
        /// 获取切线上的两个点
        /// </summary>
        /// <param name="info"></param>
        /// <param name="range"></param>
        /// <returns></returns>
        public (DataPoint p1, DataPoint p2) GetTangentLinePoints(GradientInfo info, float range)
        {

            float x1 = info.x + range;
            float y1 = info.k * (x1 - info.x) + info.y;
            DataPoint p1 = new DataPoint(x1, y1);

            float x2 = info.x - range;
            float y2 = info.k * (x2 - info.x) + info.y;
            DataPoint p2 = new DataPoint(x2, y2);

            return (p1, p2);
        }
     
        private async void ExecuteGradientCommand()
        {
            // y = x^2 +3x -8
            Func<double, double> orginal = new Func<double, double>(x => (0.5*(x * x) + (3 * x) - 8));          
            // y' = 2x +  3
            Func<double, double> d = new Func<double, double>(x => (0.5 * (2 * x) + 3));

            FunctionSeries functionSeries = new FunctionSeries(orginal, -10, 4, 0.5, "y = x^2 +3x -8")
            {
                StrokeThickness = 3,

                Color = OxyColors.YellowGreen
            };

            LeftPlotView.AddSeries(functionSeries);

            Message = "computing...";

            await Linear.GradientDescentTaskAsync(8, 1000, orginal, OnGradientChangedCallback);

            Message = "completed";
        }
      
        private void ExecuteComputeCommand()
        {

            float[] diagVector = new float[5]
            {
               2,2,2,2,2
            };

            float[,] diagMatrix = Linear.DiagonalMatrix(diagVector);

            Message = diagMatrix.Rank.ToString();

            float[,] matrixT = Linear.Dot(diagMatrix, SampleMatrix);

            LeftPlotView. UpdatePointsToPlotView(SampleMatrix);

            RightPlotView. UpdatePointsToPlotView(matrixT);

        }
      
        private void ExecuteUpdateSourceMatrixCommannd()
        {
            int row = 5;
            int col = 2;

            Random random = new Random();

            SampleMatrix = new float[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    float x = (float)random.NextDouble() * random.Next(-10, 10);
                    // double y = random.NextDouble() * random.Next(-10, 10);
                    SampleMatrix[i, j] = x;
                }
            }

            SourceMatrix = Linear.Print(SampleMatrix);
        }
       
        private void OnGradientChangedCallback(GradientInfo eventArgs)
        {

            var points = GetTangentLinePoints(eventArgs, 3);

           LeftPlotView. UpdateLineToPlotView(points.p1, points.p2);

            LeftPlotView. UpdatePointToPlotView(eventArgs.x, eventArgs.y);

            Message = eventArgs.ToString();
        }

    }
}
