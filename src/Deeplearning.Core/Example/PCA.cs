using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Example
{
    public class PCA
    {

        public PCAEventArgs Fit(Matrix source,int k) 
        {
            Matrix matrix = source;
            //1.中心化
            var centralInfo = matrix.Centralized();

            Matrix centralizedMatrix = centralInfo.matrix;
            //2.协方差矩阵
            Matrix covMatrix = centralizedMatrix.Cov();

            //3.对协方差矩阵求特征值特征向量
            EigenDecompositionEventArgs result = MatrixDecomposition.Eig(covMatrix,500, true);

            //Matrix eigens = eigResult.Eigen;

            //4.选取有效的特征值

            Matrix eigenVectors = result.eigenVectors;// result.V;//eigResult.Vectors;

            Vector[] vectors = new Vector[k];

            for (int i = 0; i < k; i++)
            {
                vectors[i] = eigenVectors.GetVector(i);
            }

           // Vector v = eigenVectors.GetVector(0);
            //vectors[1] = eigenVectors.GetVector(1);

            Matrix D = new Matrix(vectors);

            Matrix X = D.T * source;

            return new PCAEventArgs(X,D);
        }
    }
}
