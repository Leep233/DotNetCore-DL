﻿using Deeplearning.Core.Example;
using Deeplearning.Core.Math;
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

        public DelegateCommand<object> ElementaryTransformationCommand { get; set; }

        private event Action<string> DecomposeCompleted;

        public MatrixDecomposeOparetion()
        {

            OnCreate();
        }

        private void OnCreate()
        {
            Vector[] vectors = new Vector[4]
            {
                new Vector(1, 1, 1,1),
                new Vector(1, 2, 7,0),
                new Vector(1, 3, 10,0),
                new Vector(1, 5, -9,0),
            };

            Source = new Matrix(vectors);

            OrthogonalizationCommand = new DelegateCommand<object>(ExecuteOrthogonalizationCommand);

            EigenDecompositionCommand = new DelegateCommand<object>(ExecuteEigenDecompositionCommand);

            SVDDecompositionCommand = new DelegateCommand<object>(ExecuteSVDDecompositionCommand);

            PseudoInverseCommand = new DelegateCommand<object>(ExecutePseudoInverseCommand);

            PCADecompositionCommand = new DelegateCommand<object>(ExecutePCADecompositionCommand);

            ElementaryTransformationCommand = new DelegateCommand<object>(ExecuteElementaryTransformationCommand);
        }

        private void ExecuteElementaryTransformationCommand(object type)
        {
            double[,] scales = new double[3, 3] {
            {1, 2,3 },
            {3, 7, 10 },
           // {2, 5, 7 },
            {-1,-3,-4 }};

            //scales = new double[4, 5] {
            //{1,1,1,1,1 },
            //{3,2,1,1,-3 },
            //{0,1,2,2,6 },
            //{5,4,3,3,-1 }
            //};

            scales = new double[3, 3] {
            {1,3,-3 },
            {0,-1,2 },
            {2,1,3 }
            };


            scales = new double[3, 3] {
            {0,1,1 },
            {0,-1,-1 },
            {0,0,0 },           
            };

            scales = new double[3, 3] {
            {0,1,1 },
            {0,1,0 },
            {1,-1,-1 } 
            };

            Matrix matrix = new Matrix(scales);
            Matrix result = Matrix.ElementaryTransformation(matrix);// Matrix.ElementaryTransformation(matrix);

         //  result = ElementaryTransformation(result);

            // Vector vector = Test(result);

            StringBuilder builder = new StringBuilder();

            builder.AppendLine("Source Matrix");
            builder.AppendLine(matrix.ToString()) ;
            builder.AppendLine("Trans Matrix");
            builder.AppendLine(result.ToString());
            DecomposeCompleted.Invoke(builder.ToString());
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

            var result = new PCA().Fit(matrix,1);

           // MatrixDecomposition.PCA(matrix,1,1000);
        }

        public MatrixDecomposeOparetion(Action<string> onDecomposeComleted)
        {
            DecomposeCompleted = new Action<string>(onDecomposeComleted);
            OnCreate();
        }


        private void ExecutePseudoInverseCommand(object decType)
        {
            StringBuilder stringBuilder = new StringBuilder();         

            Matrix pInvMatrix = Source.PInv(1000);

            Matrix ans2 = Source * pInvMatrix * Source;

            stringBuilder.AppendLine(Source.ToString());

            stringBuilder.AppendLine(pInvMatrix.ToString());

            DecomposeCompleted.Invoke(stringBuilder.ToString());
        }

        private void ExecuteOrthogonalizationCommand(object decType)
        {    
            DecomposeCompleted.Invoke(Orthogonalization.Decompose(decType)(Source).ToString());

        }

        private void ExecuteSVDDecompositionCommand(object decType)
        {

            Matrix matrix = Source;

            SVDEventArgs result = MatrixDecomposition.SVD(matrix, 100);

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

            double[,] vectors = new double[3, 3] {
            {5,-3,2 },
            {6,-4,4 },
            {4,-4,5 }
            };

            //double[,] vectors = new double[4, 4] {
            //{1,2,3,4 },
            //{2,1,2,3 },
            //{3,2,1,2 },
            //{4,3,2,1 }
            //};
            //double[,] vectors = new double[3, 3] {
            //{2,1,1 },
            //{0,2,0 },
            //{0,-1,1 }
            //};


            //double[,] vectors = new double[3, 3] {
            //{2,-1,0 },
            //{-1,2,-1},
            //{0,-1,2 }
            //};

            Matrix matrix = new Matrix(vectors);// Source;//new Matrix(vectors); ////new Matrix(vectors); //Source;//


            var result = MatrixDecomposition.Eig(matrix, 1000);

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("========Source=========");
            sb.AppendLine(matrix.ToString());
            sb.AppendLine(result.ToString());
            sb.AppendLine("检测");
            sb.AppendLine("Matrix * EigenVectors");
            sb.AppendLine((matrix * result.eigenVectors).ToString());
            sb.AppendLine("EigenVectors * EigenMatrix");
            sb.AppendLine((result.eigenVectors * Matrix.DiagonalMatrix(result.eigens)).ToString());
            sb.AppendLine("EigenVectors * EigenVectors * EigenVectors.Inv");



            sb.AppendLine(result.Validate().ToString());

            DecomposeCompleted?.Invoke(sb.ToString());


        }


    }
}
