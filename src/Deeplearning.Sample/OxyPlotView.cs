using Deeplearning.Core.Math;
using Deeplearning.Sample.Utils;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System;

namespace Deeplearning.Sample
{


    public class OxyPlotView
    {

        public LinearAxis X_Axis { get;protected set; }
        public LinearAxis Y_Axis { get; protected set; }
        public ScatterSeries Scatter { get; protected set; }

        public LineSeries Line { get; protected set; }

        public PlotModel PlotModel { get; set; }


        public OxyPlotView(OxyColor linColor, OxyColor scatterColor,float majorStep = 0.1f,float min=-3,float max=3)
        {

            PlotModel = new PlotModel();

            //标记xy
            PlotModel.Series.Add(OxyPlotHelper.LineThroughOrigin(min, max, "x"));
            PlotModel.Series.Add(OxyPlotHelper.LineThroughOrigin(min, max, "y"));

            //显示网格
            X_Axis = OxyPlotHelper.LinearAxisWithGrid(min, max, AxisPosition.Bottom, majorStep);
            Y_Axis = OxyPlotHelper.LinearAxisWithGrid(min, max, AxisPosition.Left, majorStep);
            PlotModel.Axes.Add(X_Axis);
            PlotModel.Axes.Add(Y_Axis);

            //初始化点
            Scatter = new ScatterSeries() { MarkerType = MarkerType.Circle, MarkerFill = scatterColor };
            PlotModel.Series.Add(Scatter);

            //初始化线
            Line = new LineSeries() { MarkerType = MarkerType.Circle, LineStyle = LineStyle.Automatic, Color = linColor};
            PlotModel.Series.Add(Line);

          
        }

        

        /// <summary>
        /// 绘制线
        /// </summary>
        /// <param name="plotPosition"></param>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        internal void UpdateLineToPlotView(DataPoint p1, DataPoint p2)
        {
            Line.Points.Clear();
            Line.Points.Add(p1);
            Line.Points.Add(p2);
            UpdateView();
        }

        /// <summary>
        /// 绘制点
        /// </summary>
        /// <param name="plotPosition"></param>
        /// <param name="matrix"></param>
        internal void UpdatePointsToPlotView(float[,] matrix)
        {
            Scatter.Points.Clear();
            Scatter.Points.AddRange(OxyPlotHelper.MatrixToPoints(matrix));
            UpdateView();
        }
      

        /// <summary>
        /// 绘制点
        /// </summary>
        /// <param name="plotPosition"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public void UpdatePointToPlotView(double x, double y)
        {
            Scatter.Points.Clear();
            Scatter.Points.Add(new ScatterPoint(x, y));
            UpdateView();
        }


        public void CleanView() {
            Scatter.Points.Clear();
            Line.Points.Clear();
            UpdateView();
        }

        internal void AddSeries(Series series)
        {
            PlotModel.Series.Add(series);
           
        }

        internal void UpdateView()
        {
            PlotModel.InvalidatePlot(true);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="recMatrix"></param>
        /// <param name="axis">0:列向量，1：行向量</param>
        internal void UpdateLineToPlotView(Matrix matrix,int axis = 0)
        {
            Line.Points.Clear();

            if (axis == 0)
            {
                for (int i = 0; i < matrix.Column; i++)
                {
                    Line.Points.Add(new DataPoint(matrix[0, i], matrix[1, i]));
                }
            }
            else
            {
                for (int i = 0; i < matrix.Row; i++)
                {
                    Line.Points.Add(new DataPoint(matrix[i, 0], matrix[i, 1]));
                }
            }

            UpdateView();
        }

        /// <summary>
        /// 绘制点
        /// </summary>
        /// <param name="plotPosition"></param>
        /// <param name="matrix"></param>
        internal void UpdatePointsToPlotView(Matrix matrix,int axis=0)
        {
            Scatter.Points.Clear();
            Scatter.Points.AddRange(OxyPlotHelper.MatrixToPoints(matrix, axis));
            UpdateView();
        }
    }
}
