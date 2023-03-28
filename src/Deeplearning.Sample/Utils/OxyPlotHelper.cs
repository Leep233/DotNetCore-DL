using System.Collections.Generic;
using Deeplearning.Core.Math;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace Deeplearning.Sample.Utils
{
    public static class OxyPlotHelper
    {

        /// <summary>
        /// 获取切线上的两个点
        /// </summary>
        /// <param name="info"></param>
        /// <param name="range"></param>
        /// <returns></returns>
        public static (DataPoint p1, DataPoint p2) GetTangentLinePoints(GradientEventArgs info, float range)
        {

            float x1 = info.x + range;
            float y1 = info.k * (x1 - info.x) + info.y;
            DataPoint p1 = new DataPoint(x1, y1);

            float x2 = info.x - range;
            float y2 = info.k * (x2 - info.x) + info.y;
            DataPoint p2 = new DataPoint(x2, y2);

            return (p1, p2);
        }


        public static LineSeries LineThroughOrigin(double minimum, double maximum, string typeName = "x") {


            LineSeries series = new LineSeries() { Color = OxyColors.Black, StrokeThickness = 2,
                CanTrackerInterpolatePoints = false
            };


            switch (typeName.ToLower())
            {
                case "x":
                    series.Points.Add(new DataPoint(minimum, 0));

                    series.Points.Add(new DataPoint(maximum, 0));
                    break;
                case "y":
                    series.Points.Add(new DataPoint(0, minimum));

                    series.Points.Add(new DataPoint(0, maximum));
                    break;
            }

            return series;

        }

        public static LinearAxis LinearAxisWithGrid(float minimum, float maximum, AxisPosition axisPosition, float majorStep) {

            LinearAxis axis = new LinearAxis();
            axis.IsZoomEnabled = true;
          
            axis.Position = axisPosition;
            axis.Minimum = minimum;
            axis.Maximum = maximum;
            axis.MajorGridlineStyle = LineStyle.Solid;
            axis.MajorTickSize = 0;
            axis.MinorTickSize = 0;
            axis.MajorStep = majorStep;

            return axis;
        }

        public static IEnumerable<ScatterPoint> MatrixToPoints(float[,] matrix,int axis = 0)
        {

            if (axis == 0) 
            {
                for (int i = 0; i < matrix.GetLength(0); i++)
                {
                    var x = matrix[i, 0];

                    var y = matrix[i, 1];

                    yield return new ScatterPoint(x, y);
                }
            } else {
                for (int i = 0; i < matrix.GetLength(1); i++)
                {
                    var x = matrix[0,i];

                    var y = matrix[1,i];

                    yield return new ScatterPoint(x, y);
                }
            }

            
            yield break;
        }

        public static IEnumerable<ScatterPoint> MatrixToPoints(Matrix matrix, int axis = 0)
        {
            if (axis == 0) 
            {
                for (int j = 0; j < matrix.Column; j++)
                {

                    double x = matrix[0, j];

                    double y = matrix[1, j];

                    yield return new ScatterPoint(x, y);

                }

            }
            else 
            {
                for (int j = 0; j < matrix.Row; j++)
                {

                    double x = matrix[ j,0];

                    double y = matrix[ j,1];

                    yield return new ScatterPoint(x, y);

                }

            }
            yield break;
        }

    }
}
