﻿using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
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

        /// <summary>
        /// 正交化命令
        /// </summary>
        public DelegateCommand<object> OrthogonalizationCommand { get; set; }
        public DelegateCommand<object> SVDDecompositionCommand { get; set; }
        public DelegateCommand<object> PCADecompositionCommand { get; set; }
        public DelegateCommand<object> EigenDecompositionCommand { get; set; }
        public DelegateCommand<object> PseudoInverseCommand { get; set; }

        private event Action<string> DecomposeCompleted;

        public MatrixDecomposeOparetion()
        {

            OnCreate();
        }

        private void OnCreate()
        {
            Vector[] vectors = new Vector[4]
            {
                new  Vector(1, 1, 1,1),
                new Vector(1, 2, 7,0),
                new Vector(1, 3, 100,0),
                new Vector(1, 5, -9,0),
            };

            Source = new Matrix(vectors);

            OrthogonalizationCommand = new DelegateCommand<object>(ExecuteOrthogonalizationCommand);

            EigenDecompositionCommand = new DelegateCommand<object>(ExecuteEigenDecompositionCommand);

            SVDDecompositionCommand = new DelegateCommand<object>(ExecuteSVDDecompositionCommand);

            PseudoInverseCommand = new DelegateCommand<object>(ExecutePseudoInverseCommand);

            PCADecompositionCommand = new DelegateCommand<object>(ExecutePCADecompositionCommand);
        }

        private void ExecutePCADecompositionCommand(object type)
        {
            Vector[] vectors = new Vector[3] {

                new Vector( -1 , -1 , 0 ,  2 ,  1),
                new Vector(  2 ,  0 , 0 , -1 , -1),
                new Vector(  2 ,  0 , 1 ,  1 ,  0),

            };

            vectors = new Vector[5] {

                new Vector( -1 , -2 ),
                new Vector( -1 ,  0 ),
                new Vector(  0 ,  0 ),
                new Vector(  2 ,  1 ),
                new Vector(  0 ,  1 ),
            };

            Matrix matrix = new Matrix(vectors);    

            MatrixDecomposition.PCA(matrix,1, ChooseOrthogonalizationFunction(type));
        }

        public MatrixDecomposeOparetion(Action<string> onDecomposeComleted)
        {
            DecomposeCompleted = new Action<string>(onDecomposeComleted);
            OnCreate();
        }

        private Func<Matrix, QRResult> ChooseOrthogonalizationFunction(object decType) {
            int.TryParse(decType.ToString(), out int type);

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

        private void ExecutePseudoInverseCommand(object decType)
        {
            StringBuilder stringBuilder = new StringBuilder();         

            Matrix pInvMatrix = Source.PInv(ChooseOrthogonalizationFunction(decType));

            Matrix ans2 = Source * pInvMatrix * Source;
            stringBuilder.AppendLine(Source.ToString());
            stringBuilder.AppendLine(ans2.ToString());

            DecomposeCompleted.Invoke(stringBuilder.ToString());
        }

        private void ExecuteOrthogonalizationCommand(object decType)
        {    
            DecomposeCompleted.Invoke(ChooseOrthogonalizationFunction(decType)(Source).ToString());

        }

        private void ExecuteSVDDecompositionCommand(object decType)
        {

            Matrix matrix = Source;

            SVDResult result = MatrixDecomposition.SVD(matrix, ChooseOrthogonalizationFunction(decType));

            StringBuilder sb = new StringBuilder();

            sb.AppendLine(result.ToString());

            sb.AppendLine(matrix.ToString());

            sb.AppendLine((result.U * result.S * result.V.T).ToString());

            DecomposeCompleted?.Invoke(sb.ToString());
        }



        /// <summary>
        /// 特征值分解
        /// </summary>
        private void ExecuteEigenDecompositionCommand(object decType)
        {

            Matrix matrix = Source;//new Matrix(vectors);

            var result = MatrixDecomposition.Eig(matrix, ChooseOrthogonalizationFunction(decType), 500);

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("========Source=========");
            sb.AppendLine(matrix.ToString());
            sb.AppendLine(result.ToString());
            sb.AppendLine("========Operation=========");
            sb.AppendLine((result.Vectors * result.Eigen * result.Vectors.T).ToString());

            DecomposeCompleted?.Invoke(sb.ToString());


        }


    }
}