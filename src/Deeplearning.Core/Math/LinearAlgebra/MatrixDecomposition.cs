using Deeplearning.Core.Attributes;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Deeplearning.Core.Math.LinearAlgebra
{
    public class MatrixDecomposition
    {


        public static SVDResult SVD01(Matrix source, Func<Matrix, QRResult> decompose, int step = 500)
        {


            Matrix matrix = source;
            //step1;
            Matrix step1Matrix = matrix * matrix.T;

            Matrix step2Matrix = matrix.T * matrix;

            Debug.WriteLine("=====================");

            Debug.WriteLine(step1Matrix.ToString());



            Debug.WriteLine(step2Matrix.ToString());

            Debug.WriteLine("=====================");

            var step1EigResult = Eig(step1Matrix, decompose, step);

            int eignValueCount = step1EigResult.Eigen.Rows <= step1EigResult.Eigen.Columns ? step1EigResult.Eigen.Rows : step1EigResult.Eigen.Columns;

            Matrix D = new Matrix(eignValueCount, eignValueCount);

            for (int i = 0; i < eignValueCount; i++)
            {
                D[i, i] = MathF.Sqrt((float)step1EigResult.Eigen[i, i]);
            }

            //step1 end;
            //step2
            Debug.WriteLine(step1EigResult.ToString());

            //Debug.WriteLine("============[Matrix.T * Matrix]===========");


            Debug.WriteLine("=====================");
            var step2EigResult = Eig(step2Matrix, decompose, step);

            Matrix V = new Matrix(matrix.Columns, D.Rows);

            for (int i = 0; i < V.Columns; i++)
            {
                for (int j = 0; j < V.Rows; j++)
                {
                    V[j, i] = step2EigResult.Vectors[j, i];
                }
            }
            //  Debug.WriteLine(step2EigResult.ToString());

            Debug.WriteLine(step2EigResult.ToString());
            //step1 end;

            return new SVDResult(step1EigResult.Vectors, D, V);
        }


        public static SVDResult SVD(Matrix source, Func<Matrix, QRResult> decompose, int step = 500) 
        {

            Matrix matrix = source.T * source;

            EigResult result = Eig(matrix,decompose,step);


            Matrix v = result.Vectors;

            int r = result.Eigen.Columns;

            Matrix d = new Matrix(r, r);

            for (int i = 0; i < r; i++) 
            {
                d[i, i] = MathF.Sqrt((float)result.Eigen[i,i]);
            }

            matrix = source * source.T;

            result = Eig(matrix, decompose, step);

            r = result.Eigen.Rows;

            List<Vector> vectors = new List<Vector>();

            for (int i = 0; i < r-1; i++)
            {
               double value = result.Eigen[i, i];

               // if (value == 0||double.MinValue>=value) continue;

              vectors.Add(result.Vectors.GetVector(i));

            }

            Debug.WriteLine(result);

            Matrix u = new Matrix(vectors.ToArray());

            return new SVDResult(u, d, v);// new SVDResult(step1EigResult.Vectors,D, V);
        }


        [Completion(false)]
        public static (Matrix egin, Matrix vectors) PCA(Matrix source)
        {

            throw new NotImplementedException();
        }


        public static EigResult Eig(Matrix source, Func<Matrix,QRResult> qrDecFunction, int step = 100)
        {

            Matrix matrix = (Matrix)source.Clone();

            int k = (int)MathF.Min(matrix.Rows, matrix.Columns);

            Matrix Q = Matrix.UnitMatrix(k);

            Matrix ak = null;

            for (int i = 0; i < step; i++)
            {
                var qr = qrDecFunction.Invoke(matrix);

                matrix = qr.R * qr.Q;

                Q = Q * qr.Q;

                ak = qr.Q * qr.R;
            }

            int size = ak.Rows;

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    if (i != j)
                    {
                        ak[i, j] = 0;
                    }
                    else
                    {

                        for (int m = i - 1; m >= 0; m--)
                        {
                            if (ak[m, m] < ak[i, i])
                            {
                                double temp = ak[i, i];
                                ak[i, i] = ak[m, m];
                                ak[m, m] = temp;

                                for (int r = 0; r < Q.Rows; r++)
                                {
                                    temp = Q[r, i];
                                    Q[r, i] = Q[r, m];
                                    Q[r, m] = temp;
                                }
                            }
                        }

                    }

                }
            }
            return new EigResult(ak,  Q);
        }

    }
}
