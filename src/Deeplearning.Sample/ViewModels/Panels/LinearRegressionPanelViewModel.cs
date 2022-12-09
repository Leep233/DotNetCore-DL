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


        public OxyPlotView PlotView { get; set; }

        private string status;

        public string Status
        {
            get { return status; }
            set { status = value; RaisePropertyChanged("Status"); }
        }


        public LinearRegressionPanelViewModel()
        {
            PlotView = new OxyPlotView(OxyColors.Blue, OxyColors.Red, 1);
        }

      
        protected override void ExecuteDistributionCommand()
        {
            switch (SelectedIndex)
            {
                case 1:
                    Bayesian();
                    break;
                case 0:
                default:
                    RegularExpression();
                    break;
            }
        }

        /// <summary>
        /// 正规方程学习W
        /// </summary>
        private void RegularExpression()
        {

            IrisData irisData = App.GetIrisTrainData();

            Matrix X = irisData.Iris.Standardized().matrix; //m *n

            Matrix XT = X.T; //n*m

            Matrix XT_X = Matrix.Dot(XT, X); //n x m * m x n = nxn

            Matrix XT_X_INV = Matrix.Inv(XT_X); //nxn

            //data -> m*n;   y -> m * 4   w -> n * 4

            Matrix weight = Matrix.Dot(Matrix.Dot(XT_X_INV, XT), irisData.Y.T); // nxn * n x m = nxm * 4  x m = n x 4

            Verified(irisData.Iris, irisData.Y, weight);
        }


        private void Verified(Matrix testData,Matrix real,Matrix weight) {

            int randomIndex = new Random(DateTime.Now.GetHashCode()).Next(0, testData.Row);

            Vector x = new Vector(testData.Column);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = testData[randomIndex, i];
            }
            //n x 4    n * 1  
            Vector predictValue = Matrix.Dot(weight.T, x);

            predictValue = CommonFunctions.Softmax(predictValue);

            predictValue.Format = "P2";

            Vector realValue = real.GetVector(randomIndex);

            double loss = Loss(predictValue, realValue);

            Status = ($"第[{randomIndex + 1}]数据 => 预测值：{predictValue},真实值：{realValue},损失值:{loss}");
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

           Vector y = ProbabilityDistribution.NormalDistriution(Y, predictY, b);

            Verified(irisData.Iris, Y, weight);

        }

        private double Loss(Vector predict, Vector real) {

            int m = predict.Length;

            return Vector.Norm(predict - real);// m;
        
        }

    }
}
