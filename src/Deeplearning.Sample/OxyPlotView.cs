using Deeplearning.Sample.Utils;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Sample
{


    public class OxyPlotView
    {
        public const int Minimum = -20;

        public const int Maximum = 20;

        public const int MajorStep = 3;

        public LinearAxis X_Axis { get;protected set; }
        public LinearAxis Y_Axis { get; protected set; }
        public ScatterSeries Scatter { get; protected set; }

        public LineSeries Line { get; protected set; }

        public PlotModel PlotModel { get; set; }


        public OxyPlotView(OxyColor linColor, OxyColor scatterColor)
        {

            PlotModel = new PlotModel();
            //标记xy
            PlotModel.Series.Add(OxyPlotHelper.LineThroughOrigin(Minimum, Maximum, "x"));
            PlotModel.Series.Add(OxyPlotHelper.LineThroughOrigin(Minimum, Maximum, "y"));

            //显示网格
            X_Axis =  OxyPlotHelper.LinearAxisWithGrid(Minimum, Maximum, AxisPosition.Bottom, MajorStep);
            Y_Axis = OxyPlotHelper.LinearAxisWithGrid(Minimum, Maximum, AxisPosition.Left, MajorStep);
            PlotModel.Axes.Add(X_Axis);
            PlotModel.Axes.Add(Y_Axis);

            //初始化线
            Line = new LineSeries() { LineStyle = LineStyle.Dot, Color = linColor };
            PlotModel.Series.Add(Line);

            //初始化点
            Scatter = new ScatterSeries() { MarkerType = MarkerType.Circle, MarkerFill = scatterColor };
            PlotModel.Series.Add(Scatter);
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
    }
}
