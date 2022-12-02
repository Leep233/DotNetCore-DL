﻿using Deeplearning.Core.Extension;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Deeplearning.Core.Math.Linear
{
    public static class Algebra
    {

        public static SVDEventArgs SVD(Matrix source, int k = -1)
        {

            //1. A^T * A get matrix V
            Matrix AT_A = Matrix.Dot(source.T, source);

            EigenDecompositionEventArgs eventArgs1 = Eig(AT_A, k);

            Matrix V = eventArgs1.eigenVectors;

            Matrix A_AT = Matrix.Dot(source, source.T);

            EigenDecompositionEventArgs eventArgs2 = Eig(A_AT, k);

            Matrix U = eventArgs2.eigenVectors;

            Vector eigens = eventArgs1.eigens;

            var result = SVDFilter(U, eigens, V);

            return new SVDEventArgs(result.U, result.D, result.V);
        }

        private static (Matrix U, Matrix D, Matrix V) SVDFilter(Matrix u, Vector eigens, Matrix v, double threshold = 0.001)
        {

            for (int i = 0; i < eigens.Length; i++)
            {
                double value = eigens[i];
                eigens[i] = value == 0 ? 0 : MathF.Sqrt(MathF.Abs((float)value));
            }


            return (u, MatrixFactory.DiagonalMatrix(eigens), v);

            //List<double> list = new List<double>();

            //for (int i = eigens.Length - 1; i >= 0; i--)
            //{
            //    double value = eigens[i];

            //    value = value == 0 ? 0 : MathF.Sqrt(MathF.Abs((float)value));

            //    if (value >= threshold)
            //    {
            //        list.Insert(0, value);
            //    }
            //}
            //double[] es = list.ToArray();

            //Matrix D = MatrixFactory.DiagonalMatrix(es);

            //Matrix U = Matrix.Clip(u, 0, 0, u.Row, es.Length);

            //Matrix V = Matrix.Clip(v, 0, 0, v.Row, es.Length);

            //return (U, D, V);
        }


        /// <summary>
        /// 幂法迭代
        /// </summary>
        /// <param name="source"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static EigenDecompositionEventArgs Eig(Matrix source, int k = -1)
        {
            Matrix matrix = source;

            int vectorCount = k <= 0 ? source.Column : k;

            double[] eigens = new double[vectorCount];

            Vector[] eigVectors = new Vector[vectorCount];

            Vector eigenVector = new Vector(matrix.Column);

            for (int i = 0; i < eigenVector.Length; i++)
            {
                eigenVector[i] = i + 1;
            }

            eigenVector = Vector.Standardized(eigenVector);


            for (int i = 0; i < vectorCount; i++)
            {
                var temp = PowerIteration(eigenVector, matrix,3000);

                eigens[i] = temp.eigen;

                eigVectors[i] = temp.vector;

                //去特征 A = A-λxx^T;
                Matrix eigenMatrix = temp.eigen * temp.vector * temp.vector.T;

                matrix = matrix - eigenMatrix;           
            }

            var sortedResult = EigenFilter(eigens, eigVectors);

            EigenDecompositionEventArgs eventArgs = new EigenDecompositionEventArgs(sortedResult.eigens, sortedResult.vectors);

            return eventArgs;
        }


        /// <summary>
        /// 特征过滤
        /// </summary>
        /// <param name="eigenValues">特征值</param>
        /// <param name="vectors">特征向量</param>
        /// <param name="quantity">保留的个数 -1表示保留全部</param>
        /// <param name="threshold">绝对值小于或等于这个阈值的特征值与对应的特征向量将会被剔除</param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        private static (Vector eigens, Vector[] vectors) EigenFilter(double[] eigenValues, Vector[] vectors)
        {

            //if (eigenValues.Length > quantity)
            //{
            //    var result = Sort(eigenValues, vectors, (a, b) => MathF.Abs((float)a) < MathF.Abs((float)b));

            //    //剔除多余的特征值和特征向量

            //    Vector eigens = new Vector(quantity);

            //    Vector[] eigenVectors = new Vector[quantity];

            //    for (int i = 0; i < quantity; i++)
            //    {
            //        eigens[i] = result.eigens[i];
            //        eigenVectors[i] = result.vectors[i];
            //    }

            //    eigenValues = eigens;

            //    vectors = eigenVectors;
            //}
            // return (eigenValues, vectors);
            return Sort(eigenValues, vectors, (a, b) => a < b); //(new Vector(eigenValues), vectors);// 

        }

        /// <summary>
        /// 特征过滤
        /// </summary>
        /// <param name="eigenValues">特征值</param>
        /// <param name="vectors">特征向量</param>
        /// <param name="quantity">保留的个数 -1表示保留全部</param>
        /// <param name="threshold">绝对值小于或等于这个阈值的特征值与对应的特征向量将会被剔除</param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        private static (Vector eigens, Vector[] vectors) EigenFilter(Vector eigenValues, Vector[] vectors)
        {

            //if (eigenValues.Length > quantity)
            //{
            //    var result = Sort(eigenValues, vectors, (a, b) => MathF.Abs((float)a) < MathF.Abs((float)b));

            //    //剔除多余的特征值和特征向量

            //    Vector eigens = new Vector(quantity);

            //    Vector[] eigenVectors = new Vector[quantity];

            //    for (int i = 0; i < quantity; i++)
            //    {
            //        eigens[i] = result.eigens[i];
            //        eigenVectors[i] = result.vectors[i];
            //    }

            //    eigenValues = eigens;

            //    vectors = eigenVectors;
            //}
            // return (eigenValues, vectors);
            return Sort(eigenValues, vectors, (a, b) => a < b);

        }

        [Obsolete("特征值无法算全")]
        public static EigenDecompositionEventArgs Eig(Matrix source, Func<Matrix, QREventArgs> decompose, int k = 300)
        {

            EigenDecompositionEventArgs eigEventArgs = null;

            bool isSymmetry = Matrix.Symmetry(source);

            Matrix Ak = source;

            Matrix Q = MatrixFactory.UnitMatrix(source.Row);

            for (int i = 0; i < k; i++)
            {
                var decResult = decompose.Invoke(Ak);

                Ak = Matrix.Dot(Matrix.Dot(decResult.Q.T, Ak), decResult.Q);

                if (isSymmetry) Q = Matrix.Dot(Q, decResult.Q);
            }

            Vector eigens = Matrix.DiagonalElements(Ak);

            if (isSymmetry)
            {
                eigEventArgs = new EigenDecompositionEventArgs(eigens, Q);
            }
            else
            {
                //系数矩阵
                Matrix eigenVectors = EigenVectors(source, eigens);

                eigEventArgs = new EigenDecompositionEventArgs(eigens, eigenVectors);
            }

            var sortedResult = Sort(eigEventArgs.eigens, eigEventArgs.eigenVectors);

            return new EigenDecompositionEventArgs(sortedResult.eigens, sortedResult.vectors);
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

        /// <summary>
        /// 伪逆
        /// </summary>
        /// <param name="source"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static Matrix PInv(Matrix source)
        {

            SVDEventArgs svdResult = SVD(source);

            Matrix D = svdResult.S;
            Matrix V = svdResult.V;
            Matrix U = svdResult.U;

            Matrix S = new Matrix(V.Column, U.Column);

            int r = (int)MathF.Min(D.Row, D.Column);

            for (int i = 0; i < r; i++)
            {
                double value = D[i, i];
                if (value == 0) continue;
                S[i, i] = (1 / value);
            }

            return Matrix.Dot(Matrix.Dot(V, S), U.T);
        }

        public static (double eigen, Vector vector) PowerIteration(Vector eigenVector, Matrix matrix, int iterations = 500)
        {


            for (int i = 0; i < iterations; i++)
            {
                Vector vector = matrix * eigenVector;

                vector = Vector.Standardized(vector);

                Vector y1 = eigenVector - vector;

                double norm1 = Vector.NoSqrtNorm(y1);

                Vector y2 = eigenVector + vector;

                double norm2 = Vector.NoSqrtNorm(y2);

                if (MathF.Abs((float)norm1) <= 0.01f || MathF.Abs((float)norm2) <= 0.01f) break;

                eigenVector = vector;
            }

            double[] eigenVector_T = eigenVector.T;

            double eigenValue = eigenVector_T * matrix * eigenVector;

            return (eigenValue, eigenVector);
        }

       

        public static (Vector eigens, Matrix vectors) Sort(Vector eigens, Matrix vectors, Func<double, double, bool> conditions)
        {
            int r = eigens.Length;

            for (int i = 0; i < r - 1; i++)
            {
                for (int j = 0; j < r - 1 - i; j++)
                {
                    int nextIndex = j + 1;

                    if (conditions(eigens[j], eigens[nextIndex]))
                    {
                        double temp = eigens[j];
                        eigens[j] = eigens[nextIndex];
                        eigens[nextIndex] = temp;
                        Vector v1 = vectors.GetVector(nextIndex);
                        Vector v2 = vectors.GetVector(j);
                        vectors = Matrix.Replace(vectors, v2, nextIndex);
                        vectors = Matrix.Replace(vectors, v1, j);
                    }
                }
            }

            return (eigens, vectors);
        }

        public static (Vector eigens, Matrix vectors) Sort(Vector eigens, Matrix vectors, int order = -1)
        {
            int r = eigens.Length;

            Func<double, double, bool> conditions = (a, b) => order == -1 ? (a < b) : (a > b);

            return Sort(eigens, vectors, conditions);

        }

        public static (Vector eigens, Vector[] vectors) Sort(Vector eigens, Vector[] vectors, int order = -1)
        {
            Func<double, double, bool> conditions = (a, b) => order == -1 ? (a < b) : (a > b);

            return Sort(eigens, vectors, conditions);
        }
        public static (Vector eigens, Vector[] vectors) Sort(double[] eigens, Vector[] vectors, int order = -1)
        {
            Func<double, double, bool> conditions = (a, b) => order == -1 ? (a < b) : (a > b);

            return Sort(eigens, vectors, conditions);
        }

        public static (Vector eigens, Vector[] vectors) Sort(double[] eigens, Vector[] vectors, Func<double, double, bool> conditions)
        {
            int r = eigens.Length;

            for (int i = 0; i < r - 1; i++)
            {
                for (int j = 0; j < r - 1 - i; j++)
                {
                    int nextIndex = j + 1;

                    if (conditions(eigens[j], eigens[nextIndex]))
                    {
                        double temp = eigens[j];
                        eigens[j] = eigens[nextIndex];
                        eigens[nextIndex] = temp;
                        Vector v1 = vectors[nextIndex];
                        vectors[nextIndex] = vectors[j];
                        vectors[j] = v1;
                    }
                }
            }
            return (new Vector(eigens), vectors);
        }

        public static (Vector eigens, Vector[] vectors) Sort(Vector eigens, Vector[] vectors, Func<double, double, bool> conditions)
        {
            int r = eigens.Length;

            for (int i = 0; i < r - 1; i++)
            {
                for (int j = 0; j < r - 1 - i; j++)
                {
                    int nextIndex = j + 1;

                    if (conditions(eigens[j], eigens[nextIndex]))
                    {
                        double temp = eigens[j];
                        eigens[j] = eigens[nextIndex];
                        eigens[nextIndex] = temp;
                        Vector v1 = vectors[nextIndex];
                        vectors[nextIndex] = vectors[j];
                        vectors[j] = v1;
                    }
                }
            }
            return (eigens, vectors);
        }


        //解方程组
        private static Vector SolveLinearEquations(Matrix matrix)
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
