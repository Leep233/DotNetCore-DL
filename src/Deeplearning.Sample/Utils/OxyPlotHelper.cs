using System;
using System.Collections.Generic;
using System.Text;
using Deeplearning.Core.Math.Models;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace Deeplearning.Sample.Utils
{
    public static class OxyPlotHelper
    {

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
            axis.IsZoomEnabled = false;
            axis.Position = axisPosition;
            axis.Minimum = minimum;
            axis.Maximum = maximum;
            axis.MajorGridlineStyle = LineStyle.Solid;
            axis.MajorTickSize = 0;
            axis.MinorTickSize = 0;
            axis.MajorStep = majorStep;

            return axis;
        }

        public static IEnumerable<ScatterPoint> MatrixToPoints(float[,] matrix)
        {

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                var x = matrix[i, 0];

                var y = matrix[i, 1];

                yield return new ScatterPoint(x, y);
            }
            yield break;
        }

        public static IEnumerable<ScatterPoint> MatrixToPoints(Matrix matrix)
        {

            for (int i = 0; i < 2; i++)
            {
                var x = matrix[i, 0];

                var y = matrix[i, 1];

                yield return new ScatterPoint(x, y);
            }
            yield break;
        }

    }
}
