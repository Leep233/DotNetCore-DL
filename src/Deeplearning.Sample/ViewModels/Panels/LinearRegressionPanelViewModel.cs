using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Common;
using Deeplearning.Core.Math.Probability;
using Deeplearning.Sample.Utils;
using OxyPlot;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Deeplearning.Sample.ViewModels.Panels
{
    public class LinearRegressionPanelViewModel : DistributionBase
    {

        private int selectedIndex = 0;

        public int SelectedIndex
        {
            get { return selectedIndex; }
            set { selectedIndex = value; RaisePropertyChanged("SelectedIndex"); }
        }


        public PlotModel PlotModel { get; set; }

        private string status;

        public string Status
        {
            get { return status; }
            set { status = value; RaisePropertyChanged("Status"); }
        }

        /// <summary>
        /// 最小二乘
        /// </summary>
        public DelegateCommand OrdinaryLeastSquaresCommand { get; set; }

        public LinearRegressionPanelViewModel()
        {
            PlotModel = new PlotModel();
            OrdinaryLeastSquaresCommand = new DelegateCommand(ExecuteOrdinaryLeastSquaresCommand);
        }

        private void ExecuteOrdinaryLeastSquaresCommand()
        {
        }

        protected override void ExecuteDistributionCommand()
        {
            switch (SelectedIndex)
            {
                case 1:
                    MaximumLikelihoodEstimate();
                    break;
                case 0:
                default:
                    RegularExpression();
                    break;
            }
        }


        public void ShowIris(IrisData data) 
        {
            PlotModel.Series.Clear();

            ScatterSeries SetosaScatter = new ScatterSeries() { 
            MarkerFill = OxyColors.Blue
            
            };
            ScatterSeries VersicolorScatter = new ScatterSeries()
            {
                MarkerFill = OxyColors.CadetBlue

            };
            ScatterSeries VirginicaScatter = new ScatterSeries()
            {
                MarkerFill = OxyColors.Olive

            };

            for (int i = 0; i < data.Types.Length; i++)
            {
                ScatterPoint point = new ScatterPoint(data.Iris[i, 0], data.Iris[i, 1]);

                switch (data.Types[i])
                {                 
                    case IrisType.Setosa:
                        SetosaScatter.Points.Add(point);
                        break;
                    case IrisType.Versicolor:
                        VersicolorScatter.Points.Add(point);
                        break;
                    case IrisType.Virginica:
                        VirginicaScatter.Points.Add(point);
                        break;
                    case IrisType.Unknown:
                    default:
                        break;
                }
            }

            PlotModel.Series.Add(SetosaScatter);

            PlotModel.Series.Add(VersicolorScatter);

            PlotModel.Series.Add(VirginicaScatter);

            PlotModel.InvalidatePlot(true);

        }

        private (Matrix x, Matrix y) GetMatrixData() {

            IrisData irisData = App.GetIrisTrainData();

            ShowIris(irisData);

            Matrix matrix = irisData.Iris;


            Matrix X = new Matrix(matrix.GetVector(0));

            Matrix Y = new Matrix(matrix.GetVector(1));

            return (X,Y);

        }

        /// <summary>
        /// 正规方程学习W
        /// </summary>
        private void RegularExpression()
        {

           var data = GetMatrixData();

            Matrix X = data.x;

            Matrix Y = data.y;

            Matrix XT = X.T;

            Matrix XT_X = Matrix.Dot(XT,X);

            Matrix XT_X_INV = Matrix.Inv(XT_X); //nxn

            Matrix XT_X_INV_XT = Matrix.Dot(XT_X_INV, XT);

            Matrix weight = Matrix.Dot( XT_X_INV_XT, Y);

            int startNum = 0;
            int endNum = 10;

            Matrix x1 = new Matrix(1,1);
            x1[0, 0] = startNum;
            Matrix x2 = new Matrix(1, 1);
            x2[0, 0] = endNum;

            Matrix m1 = Matrix.Dot(weight.T, x1);
            Matrix m2 = Matrix.Dot(weight.T, x2);

            LineSeries lineSeries = new LineSeries();

            lineSeries.Points.Add(new DataPoint(startNum, m1[0, 0]));

            lineSeries.Points.Add(new DataPoint(endNum, m2[0, 0]));

            PlotModel.Series.Add(lineSeries);

            PlotModel.InvalidatePlot(true);

        }




        private void MaximumLikelihoodEstimate() 
        {
            var data = GetMatrixData();

            Matrix X = data.x;

            Matrix Y = data.y;
        }


        private void Verified(Matrix testData,Matrix real,Matrix weight) {

            int randomIndex = new Random(DateTime.Now.GetHashCode()).Next(0, testData.Row);

            Matrix matrix = new Matrix(testData.Row, 2);


            Vector start = new Vector(testData.Column);

            Vector end = new Vector(testData.Column);

            for (int j = 0; j < testData.Column; j++)
            {
                start[j] = 0;
                end[j] = 10;
            }
                       

            //n x 4    n * 1  
            Vector p1 = Matrix.Dot(weight.T, start);

            Vector p2 = Matrix.Dot(weight.T, end);

            //PlotView.UpdateLineToPlotView(new DataPoint(0, p1[0]), new DataPoint(10, p2[0]));





            //  

            // predictValue.Format = "P2";

            //  Vector realValue = real.GetVector(randomIndex);
         
        }

        /// <summary>
        /// 贝叶斯方法学习W
        /// </summary>
        /// <exception cref="NotImplementedException"></exception>
        private void Bayesian()
        {
            IrisData irisData = App.GetIrisTrainData();

            Matrix X = irisData.Iris.Standardized().matrix; //m *n

            Matrix Y = irisData.Y;

            Matrix weight = new Matrix(X.Column,3);

            Matrix predictY =  Matrix.Dot(X,weight).T; // m x n * n *4;

            Matrix b = MatrixFactory.UnitMatrix((int)MathF.Max(predictY.Row, predictY.Column));

           Vector y = Distribution.NormalDistriution(Y, predictY, b);

            Verified(irisData.Iris, Y, weight);

        }

        private Vector Loss(Matrix predict, Matrix real) {

            int m = predict.Row;

            Vector vector = new Vector(predict.Column);


            for (int i = 0; i < predict.Column; i++)
            {
                float sum = 0;

                for (int j = 0; j < m; j++)
                {
                    sum +=MathF.Pow(predict[j,i] - real[j,i],2);
                }

                vector[i] = sum/m;
            }

            return vector;
        }

    }
}
