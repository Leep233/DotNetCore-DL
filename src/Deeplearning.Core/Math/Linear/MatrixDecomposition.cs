using Deeplearning.Core.Attributes;
using Deeplearning.Core.Extension;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public class MatrixDecomposition
    {

       

        public static SVDEventArgs SVD(Matrix source, int iterations = 15)
        {
            //1. A^T * A get matrix V
            Matrix aTA = source.T * source;     

           EigenDecompositionEventArgs eventArgs1 =  Eig(aTA, iterations, true);

           Matrix V = eventArgs1.eigenVectors;

            //2.A*A^T .get Matrix.U
            Matrix aAT = source * source.T;

            EigenDecompositionEventArgs eventArgs2 = Eig(aAT, iterations, true);
       
            Matrix U = eventArgs2.eigenVectors;

            //get matrix S

            Vector eigens = eventArgs1.eigens;

            int r = U.Column;

            int c = V.Column;

            Matrix D = new Matrix(r, c);

            int l = (int)MathF.Min(r, c);

            l = (int)MathF.Min(l, eigens.Length);

            for (int i = 0; i < l; i++)
            {
                double value = eigens[i];

                value = value == 0 ? 0 : MathF.Sqrt(MathF.Abs((float)value));

                D[i, i] = value;
            }
            return new SVDEventArgs(U, D, V);
        }

   
        /// <summary>
        /// 幂法迭代
        /// </summary>
        /// <param name="source"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static EigenDecompositionEventArgs Eig(Matrix source, int k = 100, bool isClip = true)
        {
            Matrix matrix = source;

            int vectorCount = source.Column * source.Row;       

            Vector eigenValues = new Vector(vectorCount);

            Vector[] eigenVectors = new Vector[vectorCount];  

            for (int count = 0; count < vectorCount; count++)
            {
                EigenEventArgs eigenEvent = null;
                try
                {
                    eigenEvent = Linear.Algebra.PowerIteration(matrix, k);         
                }
                catch (Exception ex)
                {
                   // Debug.WriteLine(ex.Message);
                }

                eigenValues[count] = eigenEvent.eigen;

                eigenVectors[count] = eigenEvent.vector;

                //去特征 A = A-λxx^T;
                Matrix eigenMatrix = eigenValues[count] * eigenVectors[count] * eigenVectors[count].T;

                matrix = matrix - eigenMatrix;
            }

            bool isSymmetry = true;// source.Symmetry(); ////

            EigenDecompositionEventArgs eventArgs = new EigenDecompositionEventArgs(eigenValues, new Matrix(eigenVectors), isSymmetry);//.Clip(source.Column); //.Clip(source.Column);//.Sort();//
           
            return isClip? eventArgs.Clip(source.Column): eventArgs;
        }

        


        [Obsolete("特征值无法算全")]
        public static EigenDecompositionEventArgs Eig(Matrix source, Func<Matrix, QREventArgs> decompose, int k = 300)
        {

            EigenDecompositionEventArgs eigEventArgs = null;

            bool isSymmetry = source.Symmetry();

            Matrix Ak = source;

            Matrix Q = Matrix.UnitMatrix(source.Row);

            for (int i = 0; i < k; i++)
            {
                var decResult = decompose.Invoke(Ak);

                Ak = decResult.Q.T * Ak * decResult.Q;

                if (isSymmetry) Q = Q * decResult.Q;
            }

            Vector eigens = new Vector(Ak.DiagonalElements());

            if (isSymmetry)
            {
                eigEventArgs = new EigenDecompositionEventArgs(eigens, Q, true);
            }
            else
            {
                //系数矩阵
                Matrix eigenVectors = EigenVectors(source, eigens);

                eigEventArgs = new EigenDecompositionEventArgs(eigens, eigenVectors);
            }

            return eigEventArgs.Sort();
        }



        private static Matrix EigenVectors(Matrix source, Vector eigenValues)
        {
            int eigenCount = eigenValues.Length;

            Vector[] eigenVectors = new Vector[eigenCount];

            for (int i = 0; i < eigenCount; i++)
            {
                Matrix coefficientMatrix = source;

                double eigenValue = eigenValues[i];

                for (int j = 0; j < eigenCount; j++)
                {
                    double s1 = source[j, j];

                    double value = s1 - eigenValue;

                    coefficientMatrix[j, j] = value;
                }
                Matrix temp = Matrix.ElementaryTransformation(coefficientMatrix);

                eigenVectors[i] = SolveLinearEquations(temp);
            }
            return new Matrix(eigenVectors);
        }

        //解方程组
        public static Vector SolveLinearEquations(Matrix matrix)
        {

            int size = matrix.Row;

            int col = matrix.Column;

            Vector vector = new Vector(size);

            vector[size - 1] = 1;

            double mid = 1;

            //从下往上求解
            for (int r = size - 2; r >= 0; r--)
            {
                double sum = 0;

                for (int c = col - 1; c > r; c--)
                {
                    double x = vector[c];

                    double d1 = matrix[r, c];

                    sum += d1 * x;
                }

                double d2 = matrix[r, r];

                d2 = (d2 == 0) ? 0 : -sum / d2;

                vector[r] = d2;

                mid += sum * sum;
            }

            mid = MathF.Sqrt((float)mid);

            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] /= mid;
            }

            return vector;

        }

    }
}
