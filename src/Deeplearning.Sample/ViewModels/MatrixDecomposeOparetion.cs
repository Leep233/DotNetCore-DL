using Deeplearning.Core.Math.LinearAlgebra;
using Deeplearning.Core.Math.Models;
using Prism.Commands;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Sample.ViewModels
{
    public class MatrixDecomposeOparetion
    {
        public Matrix Source { get; set; }

        public DelegateCommand ClassicalGramSchmidtCommand { get; set; }
        public DelegateCommand ModifiedGramSchmidtCommand { get; set; }
        public DelegateCommand HouseholderCommand { get; set; }
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

        }
        /// <summary>
        /// 特征值分解
        /// </summary>
        private void ExecuteEigenDecompositionCommand(object decType)
        {

            int.TryParse(decType.ToString(), out int type);

            Func<Matrix, (Matrix Q, Matrix R)> function = Orthogonalization.Householder;

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

            var result = MatrixDecomposition.Eig(Source, function);

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("========Source=========");
            sb.AppendLine(Source.ToString());

            sb.AppendLine("========Eign Matrix=========");
            sb.AppendLine(result.egin.ToString());
            sb.AppendLine("========Vectors Matrix=========");
            sb.AppendLine(result.vectors.ToString());

            sb.AppendLine("========Operation=========");
            sb.AppendLine((result.vectors * result.egin * result.vectors.T).ToString());

            DecomposeCompleted?.Invoke(sb.ToString());


        }

        /// <summary>
        /// Householder QR 分解
        /// </summary>

        private void ExecuteHouseholderCommand()
        {
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
