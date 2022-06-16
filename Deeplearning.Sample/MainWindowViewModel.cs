using Deeplearning.Core.Math;
using Deeplearning.Sample.Utils;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Sample
{
    public class MainWindowViewModel : BindableBase
    {

        internal enum PlotPosition
        {
            Left,
            Right,
            All
        }

        public const int PointSize = 3;

        public const int Minimum = -20;

        public const int Maximum = 20;

        public const int MajorStep = 1;

        private double[,] SampleMatrix;

        private ScatterSeries leftScatterSeries;

        private ScatterSeries rightScatterSeries;

        private LineSeries leftLineSeries;

        private LineSeries rightLineSeries;


        public PlotModel LeftPlotViewModel { get; set; }

        public PlotModel RightPlotViewModel { get; set; }


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

        public DelegateCommand UpdateSourceMatrixCommannd { get; set; }

        public DelegateCommand ComputeCommand { get; set; }

        public DelegateCommand GradientCommand { get; set; }


        public MainWindowViewModel()
        {
            LeftPlotViewModel = new PlotModel();
            RightPlotViewModel = new PlotModel();

            #region 标注XY轴

            //标记xy
            LeftPlotViewModel.Series.Add(OxyPlotHelper.LineThroughOrigin(Minimum, Maximum, "x"));
            LeftPlotViewModel.Series.Add(OxyPlotHelper.LineThroughOrigin(Minimum, Maximum, "y"));

            RightPlotViewModel.Series.Add(OxyPlotHelper.LineThroughOrigin(Minimum, Maximum, "x"));
            RightPlotViewModel.Series.Add(OxyPlotHelper.LineThroughOrigin(Minimum, Maximum, "y"));

            #endregion

            #region 显示网格
            //显示网格
            LeftPlotViewModel.Axes.Add(OxyPlotHelper.LinearAxisWithGrid(Minimum, Maximum, AxisPosition.Bottom, MajorStep));
            LeftPlotViewModel.Axes.Add(OxyPlotHelper.LinearAxisWithGrid(Minimum, Maximum, AxisPosition.Left, MajorStep));

            RightPlotViewModel.Axes.Add(OxyPlotHelper.LinearAxisWithGrid(Minimum, Maximum, AxisPosition.Bottom, MajorStep));
            RightPlotViewModel.Axes.Add(OxyPlotHelper.LinearAxisWithGrid(Minimum, Maximum, AxisPosition.Left, MajorStep));
            //end
            #endregion

            #region 设置线
            //初始化线
            leftLineSeries = new LineSeries() { LineStyle = LineStyle.Dot, Color = OxyColors.DarkOrange};
            LeftPlotViewModel.Series.Add(leftLineSeries);

            rightLineSeries = new LineSeries() { LineStyle = LineStyle.Dot, Color = OxyColors.Olive };
            RightPlotViewModel.Series.Add(rightLineSeries);
            //end
            #endregion

            #region 设置点

            //初始化点
            leftScatterSeries = new ScatterSeries() { MarkerType = MarkerType.Circle, MarkerFill = OxyColors.Green };
            LeftPlotViewModel.Series.Add(leftScatterSeries);

            rightScatterSeries = new ScatterSeries() { MarkerType = MarkerType.Circle, MarkerFill = OxyColors.Red };
            RightPlotViewModel.Series.Add(rightScatterSeries);
            //end
            #endregion

            ExecuteUpdateSourceMatrixCommannd();

            UpdateSourceMatrixCommannd = new DelegateCommand(ExecuteUpdateSourceMatrixCommannd);

            ComputeCommand = new DelegateCommand(ExecuteComputeCommand);

            GradientCommand = new DelegateCommand(ExecuteGradientCommand);
        }

        ~MainWindowViewModel() {
            ClearPlotView( PlotPosition.All);
        }

        /// <summary>
        /// 获取切线上的两个点
        /// </summary>
        /// <param name="info"></param>
        /// <param name="range"></param>
        /// <returns></returns>
        public (DataPoint p1, DataPoint p2) GetTangentLinePoints(Gradient2D.GradientInfo info, double range)
        {

            double x1 = info.x + range;
            double y1 = info.k * (x1 - info.x) + info.y;
            DataPoint p1 = new DataPoint(x1, y1);

            double x2 = info.x - range;
            double y2 = info.k * (x2 - info.x) + info.y;
            DataPoint p2 = new DataPoint(x2, y2);

            return (p1, p2);
        }
     
        /// <summary>
        /// 清空视图
        /// </summary>
        /// <param name="plotPosition"></param>
        internal void ClearPlotView(PlotPosition plotPosition)
        {

            switch (plotPosition)
            {
                case PlotPosition.Left:
                    {
                        leftLineSeries.Points.Clear();
                        leftScatterSeries.Points.Clear();
                        LeftPlotViewModel.InvalidatePlot(true);
                    }

                    break;
                case PlotPosition.Right:
                    {
                        rightLineSeries.Points.Clear();
                        rightScatterSeries.Points.Clear();
                        RightPlotViewModel.InvalidatePlot(true);
                    }
                    break;
                case PlotPosition.All:
                    {
                        leftLineSeries.Points.Clear();
                        leftScatterSeries.Points.Clear();
                        rightLineSeries.Points.Clear();
                        rightScatterSeries.Points.Clear();
                        LeftPlotViewModel.InvalidatePlot(true);
                        RightPlotViewModel.InvalidatePlot(true);
                    }
                    break;
            }
        }

        /// <summary>
        /// 绘制线
        /// </summary>
        /// <param name="plotPosition"></param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        internal void UpdateLineToPlotView(PlotPosition plotPosition, DataPoint p1, DataPoint p2)
        {
            switch (plotPosition)
            {
                case PlotPosition.Left:
                    leftLineSeries.Points.Clear();
                    leftLineSeries.Points.Add(p1);
                    leftLineSeries.Points.Add(p2);
                    LeftPlotViewModel.InvalidatePlot(true);
                    break;
                case PlotPosition.Right:
                    rightLineSeries.Points.Clear();
                    rightLineSeries.Points.Add(p1);
                    rightLineSeries.Points.Add(p2);
                    RightPlotViewModel.InvalidatePlot(true);
                    break;
                case PlotPosition.All:
                    {
                        leftLineSeries.Points.Clear();
                        leftLineSeries.Points.Add(p1);
                        leftLineSeries.Points.Add(p2);
                        rightLineSeries.Points.Clear();
                        rightLineSeries.Points.Add(p1);
                        rightLineSeries.Points.Add(p2);
                        LeftPlotViewModel.InvalidatePlot(true);
                        RightPlotViewModel.InvalidatePlot(true);
                    }
                    break;
            }
        }

        /// <summary>
        /// 绘制点
        /// </summary>
        /// <param name="plotPosition"></param>
        /// <param name="matrix"></param>
        internal void UpdatePointsToPlotView(PlotPosition plotPosition, double[,] matrix)
        {
            switch (plotPosition)
            {
                case PlotPosition.Left:
                    {
                        leftScatterSeries.Points.Clear();
                        leftScatterSeries.Points.AddRange(OxyPlotHelper.MatrixToPoints(matrix));
                        LeftPlotViewModel.InvalidatePlot(true);
                    }
                    break;
                case PlotPosition.Right:
                    {
                        rightScatterSeries.Points.Clear();
                        rightScatterSeries.Points.AddRange(OxyPlotHelper.MatrixToPoints(matrix));
                        RightPlotViewModel.InvalidatePlot(true);
                    }
                    break;
                case PlotPosition.All:
                    {
                        leftScatterSeries.Points.Clear();
                        leftScatterSeries.Points.AddRange(OxyPlotHelper.MatrixToPoints(matrix));
                        rightScatterSeries.Points.Clear();
                        rightScatterSeries.Points.AddRange(OxyPlotHelper.MatrixToPoints(matrix));
                        LeftPlotViewModel.InvalidatePlot(true);
                        RightPlotViewModel.InvalidatePlot(true);
                    }
                    break;
            }
        }

        /// <summary>
        /// 绘制点
        /// </summary>
        /// <param name="plotPosition"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        internal void UpdatePointToPlotView(PlotPosition plotPosition, double x, double y)
        {
            switch (plotPosition)
            {
                case PlotPosition.Left:
                    {
                        leftScatterSeries.Points.Clear();
                        leftScatterSeries.Points.Add(new ScatterPoint(x, y));
                        LeftPlotViewModel.InvalidatePlot(true);
                    }
                    break;
                case PlotPosition.Right:
                    {
                        rightScatterSeries.Points.Clear();
                        rightScatterSeries.Points.Add(new ScatterPoint(x, y));
                        RightPlotViewModel.InvalidatePlot(true);
                    }
                    break;
                case PlotPosition.All:
                    {
                        leftScatterSeries.Points.Clear();
                        leftScatterSeries.Points.Add(new ScatterPoint(x, y));
                        rightScatterSeries.Points.Clear();
                        rightScatterSeries.Points.Add(new ScatterPoint(x, y));
                        LeftPlotViewModel.InvalidatePlot(true);
                        RightPlotViewModel.InvalidatePlot(true);
                    }
                    break;
            }
        }


        private async void ExecuteGradientCommand()
        {
            // y = x^2 +3x -8
            Func<double, double> orginal = new Func<double, double>(x => (0.5*(x * x) + (3 * x) - 8));          
            // y' = 2x +  3
            Func<double, double> d = new Func<double, double>(x => (0.5 * (2 * x) + 3));

            FunctionSeries functionSeries = new FunctionSeries(orginal, -10, 4, 0.5, "f(x)")
            {
                StrokeThickness = 3,

                Color = OxyColors.YellowGreen
            };

            LeftPlotViewModel.Series.Add(functionSeries);

            LeftPlotViewModel.InvalidatePlot(true);

            Gradient2D gradient2D = new Gradient2D(orginal);

            gradient2D.GradientChangedEvent += OnGradientChangedCallback;

            Message = "computing...";

            await gradient2D.GradientDescentTaskAsync(1000, 0.005d);

            Message = "completed";
        }
      
        private void ExecuteComputeCommand()
        {

            double[] diagVector = new double[5]
            {
               2,2,2,2,2
            };

            double[,] diagMatrix = Linear.DiagonalMatrix(diagVector);

            Message = diagMatrix.Rank.ToString();

            double[,] matrixT = Linear.Dot(diagMatrix, SampleMatrix);

            UpdatePointsToPlotView(PlotPosition.Left, SampleMatrix);

            UpdatePointsToPlotView(PlotPosition.Right, matrixT);

            RightPlotViewModel.InvalidatePlot(true);

            LeftPlotViewModel.InvalidatePlot(true);
        }
      
        private void ExecuteUpdateSourceMatrixCommannd()
        {
            int row = 5;
            int col = 2;

            Random random = new Random();

            SampleMatrix = new double[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    double x = random.NextDouble() * random.Next(-10, 10);
                    // double y = random.NextDouble() * random.Next(-10, 10);
                    SampleMatrix[i, j] = x;
                }
            }

            SourceMatrix = Linear.Print(SampleMatrix);
        }
       
        private void OnGradientChangedCallback(object sender, Gradient2D.GradientInfo eventArgs)
        {

            var points = GetTangentLinePoints(eventArgs, 3);

            UpdateLineToPlotView(PlotPosition.Left, points.p1, points.p2);

            UpdatePointToPlotView(PlotPosition.Left, eventArgs.x, eventArgs.y);

            Message = eventArgs.ToString();
        }

    }
}
