using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Common;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Sample.Utils;
using OxyPlot;
using OxyPlot.Series;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Deeplearning.Sample.ViewModels.Components
{
    public class MatrixOperationComponentViewModel:BindableBase
    {

        public const int VectorCount = 100;

        private Matrix poltMatrix;

        private string message;

        public string Message
        {
            get { return message; }
            set { message = value;RaisePropertyChanged("Message"); }
        }

        private int interval;

        public int Interval
        {
            get { return interval; }
            set { interval = value; RaisePropertyChanged("Interval"); }
        }

        private string status;

        public string Status
        {
            get { return status; }
            set { status = value; RaisePropertyChanged("Status"); }
        }

        public IOComponentViewModel IOComponent { get; set; }

        public OxyPlotView PlotView { get; set; }  

        public DelegateCommand UpdateMatrixCommand { get; set; }
        public DelegateCommand MatrixTransposeCommand { get; set; }

        public DelegateCommand MatrixDetCommand { get; set; }

        public DelegateCommand AdjointMatrixCommand { get; set; }

        public DelegateCommand MatrixInvCommand { get; set; }

        public DelegateCommand MatrixStandardizedCommand { get; set; }

        public DelegateCommand MatrixCentralizedCommand { get; set; }
        
        public DelegateCommand MatrixCovCommand { get; set; }

        public DelegateCommand<object> QRDecomposionCommand { get; set; }

        public DelegateCommand EigenDecomposionCommand { get; set; }
        public DelegateCommand SVDDecomposionCommand { get; set; }
        public DelegateCommand<object> PCACommand { get; set; }


        public MatrixOperationComponentViewModel()
        {
            IOComponent = new IOComponentViewModel();

            PlotView = new OxyPlotView(OxyColors.Red,OxyColors.GreenYellow);

            UpdateMatrixCommand = new DelegateCommand(ExecuteUpdateMatrixCommand);

            MatrixTransposeCommand = new DelegateCommand(ExecuteMatrixTransposeCommand);

            MatrixDetCommand = new DelegateCommand(ExecuteMatrixDetCommand);

            AdjointMatrixCommand = new DelegateCommand(ExecuteAdjointMatrixCommand);

            MatrixInvCommand = new DelegateCommand(ExecuteMatrixInvCommand);

            MatrixStandardizedCommand = new DelegateCommand(ExecuteMatrixStandardizedCommand);

            MatrixCentralizedCommand = new DelegateCommand(ExecuteMatrixCentralizedCommand);

            MatrixCovCommand = new DelegateCommand(ExecuteMatrixCovCommand);

            QRDecomposionCommand = new DelegateCommand<object>(ExecuteQRDecomposionCommand);

            EigenDecomposionCommand = new DelegateCommand(ExecuteEigenDecomposionCommand);

            SVDDecomposionCommand = new DelegateCommand(ExecuteSVDDecomposionCommand);

            PCACommand = new DelegateCommand<object>(ExecutePCACommand);

          

            UpdateMatrixCommand.Execute();

            Matrix m = new Matrix(new float[,]{ {1,2,3,4 },{2,3,4,1 },{3,4,1,2 },{4,1,2,3 } });

            IOComponent.InputContent = m.ToString();
        }

        private void ExecuteMatrixCovCommand()
        {
            Matrix matrix = GetInputMatrix();

            var eventArgs = matrix.Cov();

            IOComponent.OutputContent = eventArgs.ToString();
        }

        private void ExecuteMatrixStandardizedCommand()
        {
            Matrix matrix = GetInputMatrix();

            var eventArgs = matrix.Standardized();

            IOComponent.OutputContent = eventArgs.ToString();
        }


        private void ExecuteMatrixCentralizedCommand()
        {
            Matrix matrix = GetInputMatrix();

            var eventArgs = matrix.Centralized();

            IOComponent.OutputContent = eventArgs.ToString();
        }



        private void ExecutePCACommand(object type)
        {
            Matrix source = App.GetIrisTrainData().Iris; //

            var result = source.Standardized(0);

            Matrix standardMatrix = result.matrix;

            PlotView.UpdatePointsToPlotView(standardMatrix, 1);

            Matrix matrix;

            int k = 1;

            switch (type?.ToString().ToLower())
            {
                case "svd":
                    {

                        Vector means = result.means;                    

                       var svdEventArgs = Algebra.SVD(standardMatrix);

                        Matrix eigenVectors = svdEventArgs.V;

                        int dimension = eigenVectors.Row;

                       Matrix D = Matrix.Clip(eigenVectors, 0, 0, dimension, k);

                       Matrix X = Matrix.Dot(standardMatrix, D);

                        matrix = Matrix.Dot(X, D.T);
                      
                    }
                    break;
                case "eigen":
                default:
                    {
              

                        Vector means = result.means;

                        Matrix centeredMatrix = standardMatrix;

                        Matrix covMatrix = centeredMatrix.Cov();

                        var eigenEventArgs = Algebra.Eig(covMatrix);

                        Matrix eigenVectors = eigenEventArgs.eigenVectors;

                        int dimension = eigenVectors.Row;

                        Matrix D = Matrix.Clip(eigenVectors, 0, 0, dimension, k);

                        Matrix X = Matrix.Dot(centeredMatrix,D);

                        matrix = Matrix.Dot(X, D.T);
                    }
                    break;
            }

            PlotView.UpdateLineToPlotView(matrix,1);
        }

        private void ExecuteSVDDecomposionCommand()
        {
            Matrix matrix = GetInputMatrix();

            var eventArgs = Algebra.SVD(matrix);

            Vector eigens = Matrix.DiagonalElements(eventArgs.S);

            Vector present = CommonFunctions.Softmax(eigens);
            present.Format = "P8";

            IOComponent.OutputContent = eventArgs.ToString() + "\n" + $"{present}";

    
        }

        private void ExecuteEigenDecomposionCommand()
        {
             Matrix matrix = GetInputMatrix();
            var eventArgs =   Algebra.Eig(matrix);

            Vector present = CommonFunctions.Softmax(eventArgs.eigens);
            present.Format = "P8";

            IOComponent.OutputContent = eventArgs.ToString() + "\n" + $"{present}";
        }

        private void ExecuteQRDecomposionCommand(object type)
        {
           var decompose = Orthogonalization.Decompose(type);

            Matrix matrix = GetInputMatrix(); 

            QREventArgs eventArgs = decompose(matrix);  

            IOComponent.OutputContent = eventArgs.ToString();
        }

        private void ExecuteMatrixInvCommand()
        {
   

            try
            {
                Matrix matrix = GetInputMatrix();

                if (matrix.IsSquare) 
                {
                    IOComponent.OutputContent = Matrix.Inv(matrix).ToString();
                    Message = "逆矩阵";
                } else
                {
                    IOComponent.OutputContent = Algebra.PInv(matrix).ToString();
                    Message = "伪逆";
                }
            }
            catch (Exception ex)
            {
                Status = ex.Message;
            }

       
        }

        private void ExecuteAdjointMatrixCommand()
        {
            try
            {
                Matrix matrix = GetInputMatrix();

                if (matrix.IsSquare)
                {
                    IOComponent.OutputContent = Matrix.Adjugate(matrix).ToString();
                    Message = "伴随矩阵";
                }              
            }
            catch (Exception ex)
            {
                Status = ex.Message;
            }
        }

        private void ExecuteMatrixDetCommand()
        {
            try
            {
                Matrix matrix = GetInputMatrix();

                if (matrix.IsSquare)
                {
                    IOComponent.OutputContent = Matrix.Det(matrix).ToString();
                    Message = "行列式";
                }
            }
            catch (Exception ex)
            {
                Status = ex.Message;
            }
        }

        private void ExecuteMatrixTransposeCommand()
        {
            try
            {
                Matrix matrix = GetInputMatrix();

                IOComponent.OutputContent = matrix.T.ToString();
            }
            catch (Exception ex)
            {
                Status = ex.Message;
            }
          
        }

        private Matrix GetInputMatrix()
        {
            return TextUtil.StringToMatrix(IOComponent.InputContent);
        }



        private void ExecuteUpdateMatrixCommand()
        {
            Random random = new Random();

            float min = 0;
            float max = 1;
            float s = max- min;

            poltMatrix = new Matrix(2, VectorCount);

            for (int i = 0; i < poltMatrix.Row; i++)
            {
                for (int j = 0; j < poltMatrix.Column; j++)
                {
                    double x = random.NextDouble();

                    poltMatrix[i, j] = (float)(min + x * s);
 
                }
            }
            UpdatePlotView(poltMatrix);
        }

        private void UpdatePlotView(Matrix matrix) 
        {
            PlotView.UpdatePointsToPlotView(matrix);
        }

    }
}
