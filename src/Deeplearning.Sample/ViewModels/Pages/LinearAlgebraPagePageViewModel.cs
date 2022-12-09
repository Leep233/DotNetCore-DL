using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Core.Math.Probability;
using Deeplearning.Sample.Utils;
using Deeplearning.Sample.ViewModels.Components;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Text;

namespace Deeplearning.Sample.ViewModels.Pages
{
    public class LinearAlgebraPagePageViewModel:BindableBase
    {

        public IOComponentViewModel IOComponentContext { get; set; }
        public DelegateCommand TestCommand { get; set; }
        public DelegateCommand MatrixRandomCommand { get; set; }
        public DelegateCommand<object> SelectedQRFunctionCommand { get; set; }        
        public DelegateCommand MatrixNormalizedCommand { get; set; }
        public DelegateCommand MatrixTransposeCommand { get; set; }
        public DelegateCommand MatrixDetCommand { get; set; }
        public DelegateCommand MatrixAdjugateCommand { get; set; }
        public DelegateCommand MatrixInverseCommand { get; set; }
        public DelegateCommand VarianceMatrixCommand { get; set; }
        public DelegateCommand CovarianceMatrixCommand { get; set; }
        public DelegateCommand ElementaryTransformationCommand { get; set; }
        public DelegateCommand OrthogonalizationCommand { get; set; }
        public DelegateCommand EigenDecomposeCommand { get; set; }
        public DelegateCommand SVDDecomposeCommand { get; set; }
        public DelegateCommand PCADecomposeCommand { get; set; }
        public DelegateCommand PseudoInverseCommand { get; set; }

        private Func<Matrix, QREventArgs> decompose;

        private Matrix source;
        public LinearAlgebraPagePageViewModel()
        {
         

            IOComponentContext = new IOComponentViewModel();
      
            SelectedQRFunctionCommand = new DelegateCommand<object>(ExecuteSelectedQRFunctionCommand);

            // MatrixDetCommand = new DelegateCommand(Execute);

            MatrixRandomCommand = new DelegateCommand(ExecuteMatrixRandomCommand);
            TestCommand = new DelegateCommand(ExecuteTextCommand);
            MatrixNormalizedCommand = new DelegateCommand(ExecuteMatrixNormalizedCommand);
            MatrixTransposeCommand = new DelegateCommand(ExecuteMatrixTransposeCommand);      
            MatrixDetCommand = new DelegateCommand(ExecuteMatrixDetCommand);
            MatrixAdjugateCommand = new DelegateCommand(ExecuteMatrixAdjugateCommand);
            MatrixInverseCommand = new DelegateCommand(ExecuteMatrixInverseCommand);
            VarianceMatrixCommand = new DelegateCommand(ExecuteVarianceMatrixCommand);
            CovarianceMatrixCommand = new DelegateCommand(ExecuteCovarianceMatrixCommand);
            ElementaryTransformationCommand = new DelegateCommand(ExecuteElementaryTransformationCommand);
            OrthogonalizationCommand = new DelegateCommand(ExecuteOrthogonalizationCommand);
            EigenDecomposeCommand = new DelegateCommand(ExecuteEigenDecomposeCommand);
            SVDDecomposeCommand = new DelegateCommand(ExecuteSVDDecomposeCommand);
            PCADecomposeCommand = new DelegateCommand(ExecutePCADecomposeCommand);
            PseudoInverseCommand = new DelegateCommand(ExecutePseudoInverseCommand);

            SelectedQRFunctionCommand.Execute(2);
            MatrixRandomCommand.Execute();
        }

        private void ExecuteMatrixRandomCommand()
        {
  
            int size = 5;

            source = new Matrix(size,size);
            Random random = new Random();
            for (int r = 0; r < size; r++)
            {
                for (int c = 0; c < size; c++)
                {
                    source[r, c] = random.Next(-5,5);
                }
            }

            IOComponentContext.InputContent = source.ToString();
        }

        private Matrix GetSourceMatrix() 
        {
         return  TextUtil.StringToMatrix(IOComponentContext.InputContent);
        }

        private void ExecutePseudoInverseCommand()
        {
            Matrix matrix = GetSourceMatrix();         

            SVDEventArgs eventArgs = Algebra.SVD(matrix);

            Matrix pInv_Matrix = Algebra.PInv(matrix);

            StringBuilder sb = new StringBuilder();

            sb.AppendLine(eventArgs.ToString());

            sb.AppendLine("====Validated====");

            sb.AppendLine(eventArgs.Validate().ToString());          

            IOComponentContext.OutputContent = (pInv_Matrix.ToString());

        }

        private void ExecutePCADecomposeCommand()
        {
           // Matrix matrix = GetSourceMatrix();

            //var result = new PCA().Fit(matrix, 1);

            //IOComponentContext.OutputContent = result.ToString();
        }

        private void ExecuteSVDDecomposeCommand()
        {
            Matrix matrix = GetSourceMatrix();

            SVDEventArgs eventArgs = Algebra.SVD(matrix);

            StringBuilder sb = new StringBuilder();

            sb.AppendLine(eventArgs.ToString());

            sb.AppendLine("====Validated====");

            sb.AppendLine(eventArgs.Validate().ToString());

            IOComponentContext.OutputContent = sb.ToString();
        }

        private void ExecuteEigenDecomposeCommand()
        {
         
            Matrix matrix = GetSourceMatrix();
            StringBuilder builder = new StringBuilder();

            EigenDecompositionEventArgs eventArgs = Algebra.Eig(matrix);

           // EigenDecompositionEventArgs eventArgs = MatrixDecomposition.Eig(matrix,decompose,2000);

            if (eventArgs is null) return;

            builder.AppendLine(eventArgs.ToString());

           builder.AppendLine(eventArgs.Validate().ToString()); 

            IOComponentContext.OutputContent = builder.ToString();
        }

        private void ExecuteOrthogonalizationCommand()
        {
            Matrix matrix = GetSourceMatrix();

            StringBuilder builder = new StringBuilder();

            QREventArgs eventArgs = decompose(matrix);

            builder.AppendLine(eventArgs.ToString());

            builder.AppendLine(eventArgs.Validate().ToString());

            IOComponentContext.OutputContent = builder.ToString();
        }

        private void ExecuteElementaryTransformationCommand()
        {
            Matrix matrix = GetSourceMatrix();

            IOComponentContext.OutputContent = Matrix.ElementaryTransformation(matrix).ToString();
        }

        private void ExecuteCovarianceMatrixCommand()
        {
            int k = 2;

            Matrix matrix = GetSourceMatrix();

            Matrix centeredMatrix = matrix.Centralized(0).matrix;
            /*
2.5 2.4
0.5 0.7
2.2 2.9
1.9 2.2
3.1 3.0
2.3 2.7
2 1.6
1 1.1
1.5 1.6
1.1 0.9

             */

            string s = centeredMatrix.ToString();

            Matrix covMatrix = centeredMatrix.Cov();
            var result = Algebra.Eig(covMatrix);
            string message = result.ToString();

           Matrix D = Matrix.Clip(result.eigenVectors, 0, 0, result.eigenVectors.Row, k);

            Matrix X = Matrix.Dot(centeredMatrix,D);

            IOComponentContext.OutputContent = s + "\n" + covMatrix.ToString() + "\n" + message 
                                            + "\n" + D.ToString()
                                              + "\n" + X.ToString();
        }

        private void ExecuteVarianceMatrixCommand()
        {
           // throw new NotImplementedException();
        }

        private void ExecuteMatrixInverseCommand()
        {
            Matrix matrix = GetSourceMatrix();

            Matrix inv = Matrix.Inv(matrix);// .inverse;

            StringBuilder builder = new StringBuilder();
   
            builder.AppendLine(inv.ToString());   

            IOComponentContext.OutputContent = builder.ToString();
        }

        private void ExecuteMatrixAdjugateCommand()
        {
            IOComponentContext.OutputContent = Matrix.Adjugate(GetSourceMatrix()).ToString();
        }

        private void ExecuteMatrixDetCommand()
        {
            IOComponentContext.OutputContent = Matrix.Det(GetSourceMatrix()).ToString();
        }

        private void ExecuteMatrixTransposeCommand()
        {
            IOComponentContext.OutputContent = GetSourceMatrix().T.ToString();
        }

        private void ExecuteMatrixNormalizedCommand()
        {
            Matrix matrix = GetSourceMatrix();
            IOComponentContext.OutputContent = matrix.Standardized().ToString();
        }

        private void ExecuteSelectedQRFunctionCommand(object type)
        {
            decompose = Orthogonalization.Decompose(type);
        }

        private void ExecuteTextCommand()
        {  
            IOComponentContext.OutputContent = GetSourceMatrix().ToString();
        }
    }
}
