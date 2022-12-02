using Deeplearning.Core.Example;
using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Core.Math.Probability;
using Deeplearning.Sample.ViewModels;
using OxyPlot;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Diagnostics;
using System.IO;
using System.Text;

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
        public DelegateCommand XORNetCommand { get; set; }

        public DelegateCommand LinearRegressionTrainCommand { get; set; }
        public DelegateCommand LinearRegressionPredictCommand { get; set; }

        //去中心话
        public DelegateCommand MatrixCentralizedCommand { get; set; }
        /// <summary>
        /// 协方差矩阵
        /// </summary>
        public DelegateCommand MatrixCovCommand { get; set; }

        /// <summary>
        /// 协方差矩阵
        /// </summary>
        public DelegateCommand PCACommand { get; set; }

        public MainWindowViewModel()
        {
            LeftPlotView = new OxyPlotView(OxyColors.Orange,OxyColors.DeepPink);

            RightPlotView = new OxyPlotView(OxyColors.GreenYellow, OxyColors.DodgerBlue);          

            Decompostion = new MatrixDecomposeOparetion(OnDecomposeCompletedCallback);

            ExecuteUpdateSourceMatrixCommannd();

            XORNetCommand = new DelegateCommand(ExecuteXORNetCommand);

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

            LinearRegressionTrainCommand = new DelegateCommand(ExecuteLinearRegressionTrainCommand);
            LinearRegressionPredictCommand = new DelegateCommand(ExecuteLinearRegressionPredictCommand);

            MatrixCentralizedCommand = new DelegateCommand(ExecuteMatrixCentralizedCommand);

            MatrixCovCommand = new DelegateCommand(ExecuteMatrixCovCommand);

            PCACommand = new DelegateCommand(ExecutePCACommand);
        }

        private void ExecutePCACommand()
        {
            SampleMatrix = Matrix.MeanNormalization(SampleMatrix).matrix;

            SampleMatrix = Matrix.Centralized(SampleMatrix).matrix;

            Matrix cov = ProbabilityDistribution.Cov(SampleMatrix);
            //3.对协方差矩阵求特征值特征向量
            EigenDecompositionEventArgs result = Algebra.Eig(cov);

        }
            


        private void ExecuteMatrixCovCommand()
        {
            Matrix matrix = new Matrix(new Vector[] { 
                new Vector(-2,-1,0,1,2),
                new Vector(-4,2,0,-2,4)            
            }) ;

            matrix = SampleMatrix;// matrix.T;

            matrix = ProbabilityDistribution.Cov(matrix);

            Message = matrix.ToString();

            LeftPlotView.UpdatePointsToPlotView(Matrix.MeanNormalization(matrix).matrix);
        }

        private void ExecuteMatrixCentralizedCommand()
        {
            //  SampleMatrix = Matrix.Normalized(SampleMatrix);

            // var result  = Matrix.Centralized(SampleMatrix);

            //var result = Matrix.MeanNormalization(SampleMatrix);

            //  

            SampleMatrix = ProbabilityDistribution.MeanNormalization(SampleMatrix);// ProbabilityDistribution.MinMaxScaler(SampleMatrix);// result.matrix;

           // Message = result.avgs.ToString();

            LeftPlotView.UpdatePointsToPlotView(SampleMatrix);
        }

        private LinearRegression linearRegression = new LinearRegression();
        private void ExecuteLinearRegressionPredictCommand()
        {
            //train
            // string path = "./resources/testdata/taxi-fare-train.csv";
            string path = "./resources/testdata/taxi-fare-test.csv";

            var trainData = ReadLinearRegressionData(path,10);

            Vector testData = linearRegression.Predice(trainData.data);

            double loss = InformationTheory.MES(testData, trainData.real); 

            Message = $"误差：{loss}" + "\n" +
                $"预测值：{ testData}" + "\n" +
                 $"实际值：{  trainData.real}";

        }

    
        private (Matrix data, Vector real) ReadLinearRegressionData(string path, int count)
        {
           
            string [] lines = File.ReadAllLines(path);
            
            int dataCount = count>0?count: lines.Length-1;//count;
            Vector real = new Vector(dataCount);
            int fc = lines[0].Split(',').Length-2;
            Matrix data = new Matrix(dataCount,fc);

            for (int i = 0; i < dataCount; i++)
            {
                string content = lines[i + 1];
               string[] words = content.Split(',');
                data[i,0] = 1;
                data[i,1] = float.Parse(words[1]);
                data[i,2] = float.Parse(words[2]);
                data[i,3] = float.Parse(words[3]);
                data[i,4] = float.Parse(words[4]);
                real[i] = float.Parse(words[6]);
      
            }
            return (data, real);
        }

        private  void ExecuteLinearRegressionTrainCommand()
        {
            string path = "./resources/testdata/taxi-fare-train.csv";

             var trainData = ReadLinearRegressionData(path,-1);

            double loss = linearRegression.Fit(trainData.data, trainData.real);

            StringBuilder sb = new StringBuilder();

            sb.AppendLine($"训练完成(误差:{loss})");

           Message = sb.ToString();
        }

    

        private void ExecuteXORNetCommand()
        {
            XORNet net = new XORNet();

            //Matrix w = new Matrix(2,2);

            float lr = 0.001f;
            float step = 100;
            float e = 10E-8f;

            Matrix netT = net.transData.T;

            Matrix netMatrix = Matrix.Dot(netT, net.transData);

            Matrix matrix = Matrix.Inv(netMatrix);

            Matrix m = Matrix.Dot(matrix, netT);

            Vector w = m * net.realModel;

            Message = w.ToString();

        }

        private void ExecuteMatrixNormalizedCommand()
        {
           Matrix normalMatrix = Matrix.MeanNormalization(SampleMatrix).matrix;

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
            stringBuilder.AppendLine($"Var1={ ProbabilityDistribution.Var(x, p, RandomVariableType.Discrete)}");
            stringBuilder.AppendLine($"Var2={ ProbabilityDistribution.Var(x, p2, RandomVariableType.Discrete)}");

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

            matrix = ProbabilityDistribution.Cov(matrix);

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


            //matrix = Matrix.Var(matrix);

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
            
            Matrix matrixInv = Matrix.Inv(matrix);     
            Vector v = matrix * sourceVector;
            sb.AppendLine((v).ToString());
            Vector v2 = matrixInv * v;
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


            double[,] scalars = new double[3, 3] {
            { 1,1,1},
            { 2,1,3},
            { 1,1,4}
            };

            Matrix matrix = new Matrix(scalars);

           StringBuilder sb = new StringBuilder();
            sb.AppendLine("========Sources========");
            sb.AppendLine(matrix.ToString());
            sb.AppendLine("========伴随矩阵========");
            matrix = Matrix.Adjugate(matrix);
            sb.AppendLine(matrix.ToString());
          
            Message = sb.ToString();
        }

        private void ExecuteMatrixDetCommand()
        {
            double[,] scalars = new double[3, 3] {
            {6,1,1 },
            {4,-2,5 },
            {2,8,7 }
            };

            Matrix matrix = new Matrix(scalars);

            Message = Matrix.Det(matrix).ToString("F4");
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

            Vector minVector = Gradient.GradientDescent(original,Vector.Random(2,-5,5), GradientParams.Default);

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
        public (DataPoint p1, DataPoint p2) GetTangentLinePoints(GradientEventArgs info, float range)
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

            Vector vector = await Gradient.GradientDescent(orginal,5, GradientParams.Default, OnGradientChangedCallback);

            Message = $"completed(min:{vector})";
        }
      
        private void ExecuteComputeCommand()
        {

            Vector v = new Vector(2,2,2,2,2);
 
            Matrix diagMatrix = MatrixFactory.DiagonalMatrix(v);

            Matrix matrixT = Matrix.Dot(diagMatrix , SampleMatrix);

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
       
        private void OnGradientChangedCallback(GradientEventArgs eventArgs)
        {

           var points = GetTangentLinePoints(eventArgs, 3);

           LeftPlotView. UpdateLineToPlotView(points.p1, points.p2);

           LeftPlotView. UpdatePointToPlotView(eventArgs.x, eventArgs.y);

           Message = eventArgs.ToString();
        }

    }
}
