using Deeplearning.Core.Attributes;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.LinearAlgebra
{
    public class MatrixDecomposition
    {

        [Completion(false)]
        public static (Matrix egin, Matrix vectors) SVD(Matrix source) { 
        
        throw new NotImplementedException();
        }


        [Completion(false)]
        public static (Matrix egin, Matrix vectors) PCA(Matrix source)
        {

            throw new NotImplementedException();
        }


        public static (Matrix egin, Matrix vectors) Eig(Matrix source, Func<Matrix, (Matrix Q, Matrix R)> qrDecFunction, int step = 100)
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
            return (egin: ak, vectors: Q);
        }

    }
}
