using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Models;
using Deeplearning.Sample.Utils;
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
        List<Gradient3DInfo> v3Points = new List<Gradient3DInfo>();

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
        public DelegateCommand ClassicalGramSchmidtCommand { get; set; }
        public DelegateCommand ModifiedGramSchmidtCommand { get; set; }
        public DelegateCommand HouseholderCommand { get; set; }
        public DelegateCommand EigenDecompositionCommand{ get; set; }

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

            MatrixDetCommand = new DelegateCommand(ExecuteMatrixDetCommand);

            MatrixAdjugateCommand = new DelegateCommand(ExecuteMatrixAdjugateCommand);

            ClassicalGramSchmidtCommand = new DelegateCommand(ExecuteClassicalGramSchmidtCommand);

            ModifiedGramSchmidtCommand = new DelegateCommand(ExecuteModifiedGramSchmidtCommand);

            HouseholderCommand = new DelegateCommand(ExecuteHouseholderCommand);

            EigenDecompositionCommand = new DelegateCommand(ExecuteEigenDecompositionCommand);

            TransposeCommand = new DelegateCommand(ExecuteTransposeCommand);
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

        /// <summary>
        /// 特征值分解
        /// </summary>
        private  void ExecuteEigenDecompositionCommand()
        {

            Vector[] vectors = new Vector[4];
            vectors[0] = new Vector(1, 2, 3, 4);
            vectors[1] = new Vector(2, 1, 2, 3);
            vectors[2] = new Vector(3, 2, 1, 2);
            vectors[3] = new Vector(4, 3, 2, 1);
            Matrix source = new Matrix(vectors);
    

           var result = LinearAlgebra.Eig(source,LinearAlgebra.MGS);

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("========Source=========");
            sb.AppendLine(source.ToString());

            sb.AppendLine("========Eign Matrix=========");
            sb.AppendLine(result.egin.ToString());
            sb.AppendLine("========Vectors Matrix=========");
            sb.AppendLine(result.vectors.ToString());

            sb.AppendLine("========Operation=========");
            sb.AppendLine((result.vectors * result.egin* result.vectors.T).ToString());

            Message = sb.ToString();
        }

        /// <summary>
        /// Householder QR 分解
        /// </summary>

        private void ExecuteHouseholderCommand()
        {
            Vector[] vectors = new Vector[3];
            vectors[0] = new Vector(1, -1, 0);
            vectors[1] = new Vector(2, 0, -2);
            vectors[2] = new Vector(3, -3, 3);


            Matrix matrix = new Matrix(vectors);


            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============[Householder]=========");
            var result = LinearAlgebra.Householder(matrix);
            sb.AppendLine(result.Q?.ToString());
            sb.AppendLine(result.R?.ToString());

            Message = sb.ToString();
        }

        /// <summary>
        /// ModifiedGramSchmidt 分解
        /// </summary>
        private void ExecuteModifiedGramSchmidtCommand()
        {
            ////Vector[] vectors = new Vector[3];
            //vectors[0] = new Vector(1, -1, 0);
            //vectors[1] = new Vector(2, 0, -2);
            //vectors[2] = new Vector(3, -3, 3);

            Vector[] vectors = new Vector[4];
            vectors[0] = new Vector(1, 1, 1, 1);
            vectors[1] = new Vector(2, 1, 1, 1);
            vectors[2] = new Vector(3, 2, 1, 1);
            vectors[3] = new Vector(4, 3, 2, 1);

            Matrix matrix = new Matrix(vectors);


            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============[MGS]==========");
            var result = LinearAlgebra.MGS(matrix);
            sb.AppendLine(result.Q.ToString());
            sb.AppendLine(result.R.ToString());

            Message = sb.ToString();
        }
        /// <summary>
        /// Classical Gram-Schmidt 分解
        /// </summary>
        private void ExecuteClassicalGramSchmidtCommand()
        {

            //Vector [] vectors = new Vector[3];
            //vectors[0] = new Vector(1,-1,0);
            //vectors[1] = new Vector(2,0,-2);
            //vectors[2] = new Vector(3, -3, 3);


            Vector[] vectors = new Vector[4];
            vectors[0] = new Vector(1, 1, 1, 1);
            vectors[1] = new Vector(2, 1, 1, 1);
            vectors[2] = new Vector(3, 2, 1, 1);
            vectors[3] = new Vector(4, 3, 2, 1);


            Matrix matrix = new Matrix(vectors);      

  
            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============[CGS]=========");
            var result = LinearAlgebra.CGS(matrix);
            sb.AppendLine(result.Q.ToString());
            sb.AppendLine(result.R.ToString()); 

            Message = sb.ToString();
        }

        private void ExecuteMatrixAdjugateCommand()
        {
            double[,] scalars = new double[2, 2] {
            { 4,1},
            { 3,2}};

            scalars = new double[3, 3] {
            { 2,3,1},
            { 3,4,1},
            { 3,7,2}
            };

            scalars = new double[3, 3] {
            { 2,2,1},
            { -2,1,2},
            { 1,-2,2}
            };

            Matrix matrix = new Matrix(scalars);

            matrix = matrix* matrix.inverse;

            Message = matrix.ToString();
        }

        private void ExecuteMatrixDetCommand()
        {
            double[,] scalars = new double[3, 3] {
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

            await LinearAlgebra.GradientDescentTaskAsync(500, original,OnGradient3DChangedCallback);

            using (StreamWriter writer = File.CreateText("point.txt"))
            {
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

            await LinearAlgebra.GradientDescentTaskAsync(8, 1000, orginal, OnGradientChangedCallback);

            Message = "completed";
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
            int row = 3;
            int col = 3;

            Random random = new Random();

            SampleMatrix = new Matrix(row, col);

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    float x = (float)random.NextDouble() * random.Next(-10, 10);
                    // double y = random.NextDouble() * random.Next(-10, 10);
                    SampleMatrix[i, j] = x;
                }
            }

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
