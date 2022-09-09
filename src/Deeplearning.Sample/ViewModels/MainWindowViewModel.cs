using Deeplearning.Core.Math;
using Deeplearning.Core.Math.LinearAlgebra;
using Deeplearning.Core.Math.Models;
using Deeplearning.Core.Math.Probability;
using Deeplearning.Sample.Utils;
using Deeplearning.Sample.ViewModels;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Sample
{
    public class MainWindowViewModel : BindableBase
    {
        public const int SampleMatrixRow = 2;

        public const int SampleMatrixColumn = 30;

        private Matrix SampleMatrix;

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
        public DelegateCommand TransposeCommand{get;set;}
        public DelegateCommand GradientCommand { get; set; }
        public DelegateCommand Gradient3DCommand { get; set; }
        public DelegateCommand NormalDistriutionCommand { get; set; }
        public DelegateCommand MatrixDetCommand { get; set; }
        public DelegateCommand MatrixAdjugateCommand { get; set; }
        public DelegateCommand MatrixInverseCommand { get; set; }
        public MatrixDecomposeOparetion Decompostion { get; set; }  
        public DelegateCommand VarianceMatrixCommand { get; set; }
        public DelegateCommand CovarianceMatrixCommand { get; set; }
        public DelegateCommand MatrixNormalizedCommand { get; set; } 
        public DelegateCommand TestCommand { get; set; }
        public MainWindowViewModel()
        {
            LeftPlotView = new OxyPlotView(OxyColors.Orange,OxyColors.DeepPink);

            RightPlotView = new OxyPlotView(OxyColors.GreenYellow, OxyColors.DodgerBlue);          

            Decompostion = new MatrixDecomposeOparetion(OnDecomposeCompletedCallback);

            ExecuteUpdateSourceMatrixCommannd();

            VarianceMatrixCommand = new DelegateCommand(ExecuteVarianceMatrixCommand);

            CovarianceMatrixCommand = new DelegateCommand(ExecuteCovarianceMatrixCommand);

            UpdateSourceMatrixCommannd = new DelegateCommand(ExecuteUpdateSourceMatrixCommannd);

            ComputeCommand = new DelegateCommand(ExecuteComputeCommand);

            GradientCommand = new DelegateCommand(ExecuteGradientCommand);

            Gradient3DCommand = new DelegateCommand(ExecuteGradient3DCommand);

            NormalDistriutionCommand = new DelegateCommand(ExecuteNormalDistriutionCommand);

            MatrixDetCommand = new DelegateCommand(ExecuteMatrixDetCommand);

            MatrixAdjugateCommand = new DelegateCommand(ExecuteMatrixAdjugateCommand);

            TransposeCommand = new DelegateCommand(ExecuteTransposeCommand);

            MatrixInverseCommand = new DelegateCommand(ExecuteMatrixInverseCommand);

            TestCommand = new DelegateCommand(ExecuteTestCommand);

            MatrixNormalizedCommand = new DelegateCommand(ExecuteMatrixNormalizedCommand);
  
        }

        private void ExecuteMatrixNormalizedCommand()
        {
           Matrix normalMatrix = SampleMatrix.Normalized();

            LeftPlotView.UpdatePointsToPlotView(normalMatrix);

           Message = normalMatrix.ToString();
        }

        private void ExecuteTestCommand()
        {

            StringBuilder stringBuilder = new StringBuilder();  



            Vector x = new Vector(10, 9, 8);
            Vector p = new Vector(0.1f, 0.8f, 0.1f);
            Vector p2 = new Vector(0.3f, 0.4f, 0.3f);
            stringBuilder.AppendLine($"exp={ ProbabilityDistribution.Exp(x, p)}");
            stringBuilder.AppendLine($"Var1={ ProbabilityDistribution.Var(x, p, ProbabilityDistributionMode.Discrete)}");
            stringBuilder.AppendLine($"Var2={ ProbabilityDistribution.Var(x, p2, ProbabilityDistributionMode.Discrete)}");

            x = new Vector(5, 20, 40, 80, 100);
            Vector y = new Vector(10, 24, 33, 54, 10);


            stringBuilder.AppendLine($"Cov={ ProbabilityDistribution.Cov(x, y)}"); 

            Message = stringBuilder.ToString();
        }

        private void ExecuteCovarianceMatrixCommand()
        {
            Vector[]vectors = new Vector[3]
                { 
                new Vector(-1,1),
                new Vector(0.5f,-0.5f),  
                 new Vector(1,-1),
                };

            Matrix matrix = new Matrix(vectors);

            matrix = matrix.Cov();

            Message = matrix.ToString();
        }

        private void ExecuteVarianceMatrixCommand()
        {
            Vector[] vectors = new Vector[3]
            {
                new Vector(1,3),
                new Vector(2,1),
                new Vector(3,1)
            };

            Matrix matrix = new Matrix(vectors);


            matrix = matrix.Var();

            Message = matrix.ToString();
        }

        private void ExecuteMatrixInverseCommand()
        {
           Vector [] vectors = new Vector [3];
           vectors[0] = new Vector(1,2,3);
           vectors[1] = new Vector (2,2,4);
           vectors[2] = new Vector (3,1,3);

            Matrix matrix = new Matrix(vectors);

            Vector sourceVector = new Vector(-1,2,-3);


            StringBuilder sb = new StringBuilder();
            sb.AppendLine("========Sources========");
            sb.AppendLine(matrix.ToString());
            sb.AppendLine("========逆矩阵========");     
            sb.AppendLine(matrix.inverse.ToString());
            sb.AppendLine("========检测========");
            sb.AppendLine((matrix * matrix.inverse).ToString());
            sb.AppendLine("========检测========");
            sb.AppendLine(sourceVector.ToString());
            Vector v = matrix * sourceVector;
            sb.AppendLine((v).ToString());
            Vector v2 = matrix.inverse * v;
            sb.AppendLine(v2.ToString());

            Message = sb.ToString();
        }

        private void OnDecomposeCompletedCallback(string message)
        {
            Message = message;
        }

        private void ExecuteTransposeCommand()
        {
            Vector[] vectors = new Vector[4];
            vectors[0] = new Vector(1, -1, 0);
            vectors[1] = new Vector(2, 0, -2);
            vectors[2] = new Vector(3, -3, 3);
            vectors[3] = new Vector(1, -1, 1);

            Matrix matrix = new Matrix(vectors);

            Matrix t_Matrix = matrix.T;
            StringBuilder sb = new StringBuilder();

            sb.AppendLine(matrix.ToString());
            sb.AppendLine(t_Matrix.ToString());

            Message = sb.ToString();

        }

  
        private void ExecuteMatrixAdjugateCommand()
        {


            float[,] scalars = new float[3, 3] {
            { 1,1,1},
            { 2,1,3},
            { 1,1,4}
            };

            Matrix matrix = new Matrix(scalars);

           StringBuilder sb = new StringBuilder();
            sb.AppendLine("========Sources========");
            sb.AppendLine(matrix.ToString());
            sb.AppendLine("========伴随矩阵========");
            matrix = matrix.abj;
            sb.AppendLine(matrix.ToString());
          
            Message = sb.ToString();
        }

        private void ExecuteMatrixDetCommand()
        {
            float[,] scalars = new float[3, 3] {
            {6,1,1 },
            {4,-2,5 },
            {2,8,7 }
            };

            Matrix matrix = new Matrix(scalars);

            Message = matrix.det.ToString("F4");
        }

        private void ExecuteNormalDistriutionCommand()
        {
            double dx = 0.5f;

            FunctionSeries series1 = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, 0.5f, 0.5f), -3, 3, dx)
            {
                Color = OxyColors.Red,
                Title = "正态分布(u=0.5,a=0.5)",
            };



            FunctionSeries series2 = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, 1, 0.5f), -3, 3, dx)
            {
                Color = OxyColors.Orange,
                Title = "正态分布(u=1,a=0.5f)"
            };

            FunctionSeries series3 = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, 0.5f, 1f), -3, 3, dx)
            {
                Color = OxyColors.DeepSkyBlue,
                Title = "正态分布(u=0.5,a=1f)"
            };

            FunctionSeries series4 = new FunctionSeries(x => ProbabilityDistribution.NormalDistriution((float)x, 0, 1), -3, 3, dx)
            {
                Color = OxyColors.Green,
                Title = "标准正态分布(u=0,a=1)"
            };


            LeftPlotView.AddSeries(series1);
            LeftPlotView.AddSeries(series2);
            LeftPlotView.AddSeries(series3);
            LeftPlotView.AddSeries(series4);

            LeftPlotView.UpdateView();
        }

        private async void ExecuteGradient3DCommand()
        {
            Func<Vector, float> original = new Func<Vector, float>(vector => MathF.Pow((float)vector[0], 2) + MathF.Pow((float)vector[1], 2));

            Vector minVector = Gradient.GradientDescent(original,Vector.Random(2,-5,5));

            Message = $"done...({minVector})";
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

            Vector vector = await Gradient.GradientDescent(orginal,5, OnGradientChangedCallback);

            Message = $"completed(min:{vector})";
        }
      
        private void ExecuteComputeCommand()
        {

            Vector v = new Vector(2,2,2,2,2);
 
            Matrix diagMatrix = Matrix.DiagonalMatrix(v);

            Matrix matrixT = diagMatrix * SampleMatrix;

            LeftPlotView.UpdatePointsToPlotView(SampleMatrix);

            RightPlotView.UpdatePointsToPlotView(matrixT);

        }
      
        private void ExecuteUpdateSourceMatrixCommannd()
        {

            Random random = new Random();

            SampleMatrix = new Matrix(SampleMatrixRow, SampleMatrixColumn);

            for (int i = 0; i < SampleMatrixRow; i++)
            {
                for (int j = 0; j < SampleMatrixColumn; j++)
                {
                    float x = (float)random.NextDouble() * random.Next(0, 20);
                    // double y = random.NextDouble() * random.Next(-10, 10);
                    SampleMatrix[i, j] = x;
                }
            } 

            LeftPlotView.UpdatePointsToPlotView(SampleMatrix);

            SourceMatrix = SampleMatrix.ToString();
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
