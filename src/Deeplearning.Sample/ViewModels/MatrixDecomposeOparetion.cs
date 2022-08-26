using Deeplearning.Core.Math.LinearAlgebra;
using Deeplearning.Core.Math.Models;
using Prism.Commands;
using System;
using System.Diagnostics;
using System.Text;

namespace Deeplearning.Sample.ViewModels
{
    public class MatrixDecomposeOparetion
    {
        public Matrix Source { get; set; }

        public DelegateCommand ClassicalGramSchmidtCommand { get; set; }
        public DelegateCommand ModifiedGramSchmidtCommand { get; set; }
        public DelegateCommand HouseholderCommand { get; set; }

        public DelegateCommand<object> SVDDecompositionCommand { get; set; }        
        public DelegateCommand<object> EigenDecompositionCommand { get; set; }

        private event Action<string> DecomposeCompleted;

        public MatrixDecomposeOparetion()
        {

            Vector[] vectors = new Vector[4];
            vectors[0] = new Vector(1, 2, 3, 4);
            vectors[1] = new Vector(2, 1, 2, 3);
            vectors[2] = new Vector(3, 2, 1, 2);
            vectors[3] = new Vector(4, 3, 2, 1);

            Source = new Matrix(vectors);

            ClassicalGramSchmidtCommand = new DelegateCommand(ExecuteClassicalGramSchmidtCommand);

            ModifiedGramSchmidtCommand = new DelegateCommand(ExecuteModifiedGramSchmidtCommand);

            HouseholderCommand = new DelegateCommand(ExecuteHouseholderCommand);

            EigenDecompositionCommand = new DelegateCommand<object>(ExecuteEigenDecompositionCommand);

            SVDDecompositionCommand = new DelegateCommand<object>(ExecuteSVDDecompositionCommand);


        }

        private void ExecuteSVDDecompositionCommand(object decType)
        {

            int.TryParse(decType.ToString(), out int type);


            Vector[] vectors = new Vector[] {
                new Vector(2,0,1),
                new Vector(0,2,1),
            };

            Matrix matrix = new Matrix(vectors);// (Matrix)Source.Clone();// new Matrix(vectors);

            var result = MatrixDecomposition.SVD(matrix, GetOrthogonalizationFunction(type));

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("========Source=========");
            sb.AppendLine(Source.ToString());
            sb.AppendLine("========SVD=========");
            sb.AppendLine(result.ToString());

            sb.AppendLine("========检测=========");

            sb.AppendLine((result.U * result.D * result.V.T).ToString());

            DecomposeCompleted?.Invoke(sb.ToString());
        }

        public MatrixDecomposeOparetion(Action<string> onDecomposeComleted)
        {
            DecomposeCompleted = new Action<string>(onDecomposeComleted);


            Vector[] vectors = new Vector[4];
            vectors[0] = new Vector(1, 2, 3, 4);
            vectors[1] = new Vector(2, 1, 2, 3);
            vectors[2] = new Vector(3, 2, 1, 2);
            vectors[3] = new Vector(4, 3, 2, 1);

            Source = new Matrix(vectors);

            ClassicalGramSchmidtCommand = new DelegateCommand(ExecuteClassicalGramSchmidtCommand);

            ModifiedGramSchmidtCommand = new DelegateCommand(ExecuteModifiedGramSchmidtCommand);

            HouseholderCommand = new DelegateCommand(ExecuteHouseholderCommand);

            EigenDecompositionCommand = new DelegateCommand<object>(ExecuteEigenDecompositionCommand);

            SVDDecompositionCommand = new DelegateCommand<object>(ExecuteSVDDecompositionCommand);

        }


        private Func<Matrix, QRResult> GetOrthogonalizationFunction(int type) {
            Func<Matrix, QRResult> function = Orthogonalization.Householder;

            switch (type)
            {
                case 0:
                    function = Orthogonalization.CGS;
                    break;
                case 1:
                    function = Orthogonalization.MGS;
                    break;
                case 2:
                default:
                    function = Orthogonalization.Householder;
                    break;
            }

            return function;    
        }

        /// <summary>
        /// 特征值分解
        /// </summary>
        private void ExecuteEigenDecompositionCommand(object decType)
        {

            int.TryParse(decType.ToString(), out int type);
            

            var result = MatrixDecomposition.Eig(Source, GetOrthogonalizationFunction(type));

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("========Source=========");
            sb.AppendLine(Source.ToString());     
            sb.AppendLine(result.ToString());
            sb.AppendLine("========Operation=========");
            sb.AppendLine((result.Vectors * result.Eigen * result.Vectors.T).ToString());

            DecomposeCompleted?.Invoke(sb.ToString());


        }

        /// <summary>
        /// Householder QR 分解
        /// </summary>

        private void ExecuteHouseholderCommand()
        {

            Vector[] vectors = new Vector[3];
            vectors[0] = new Vector(2,1,1);
            vectors[1] = new Vector(1,1,0);
            vectors[2] = new Vector(1,0,1);

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============[Householder]=========");
            var result = Orthogonalization.Householder(Source);
            sb.AppendLine(result.Q?.ToString());
            sb.AppendLine(result.R?.ToString());
            sb.AppendLine("============[source]=========");
            sb.AppendLine(Source.ToString());
            sb.AppendLine("============[Check]=========");
            sb.AppendLine((result.Q * result.R).ToString());

            DecomposeCompleted?.Invoke(sb.ToString());

        }

        /// <summary>
        /// ModifiedGramSchmidt 分解
        /// </summary>
        private void ExecuteModifiedGramSchmidtCommand()
        {

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============[MGS]==========");
            var result = Orthogonalization.MGS(Source);
            sb.AppendLine("========= Q ===========");
            sb.AppendLine(result.Q.ToString());
            sb.AppendLine("========= R ===========");
            sb.AppendLine(result.R.ToString());
            sb.AppendLine("============[source]=========");
            sb.AppendLine(Source.ToString());
            sb.AppendLine("============[Check]=========");
            sb.AppendLine((result.Q * result.R).ToString());

            DecomposeCompleted?.Invoke(sb.ToString());
            //  Message = sb.ToString();
        }
        /// <summary>
        /// Classical Gram-Schmidt 分解
        /// </summary>
        private void ExecuteClassicalGramSchmidtCommand()
        {       

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============[CGS]=========");
            var result = Orthogonalization.CGS(Source);
            sb.AppendLine(result.Q.ToString());
            sb.AppendLine(result.R.ToString());
            sb.AppendLine("============[source]=========");
            sb.AppendLine(Source.ToString());
            sb.AppendLine("============[Check]=========");
            sb.AppendLine((result.Q * result.R).ToString());
            DecomposeCompleted?.Invoke(sb.ToString());
        }
    }
}
