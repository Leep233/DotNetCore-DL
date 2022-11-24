using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;

namespace Deeplearning.Core.Example
{
    public class PCA
    {

        public PCAEventArgs EigFit(Matrix source,int k) 
        {
            Matrix matrix = source;
            //1.中心化
            var centralInfo = Matrix.Centralized(matrix);

            Matrix centralizedMatrix = centralInfo.matrix;
            //2.协方差矩阵
            Matrix covMatrix = Matrix.Cov(centralizedMatrix);

            //3.对协方差矩阵求特征值特征向量
            EigenDecompositionEventArgs result = Math.Linear.Algebra.Eig(covMatrix,k);

            //Matrix eigens = eigResult.Eigen;

            //4.选取有效的特征值

          //  Matrix eigenVectors = result.eigenVectors;

            Matrix D = result.eigenVectors;// eigenVectors;// Matrix.Clip(eigenVectors, 0,0, eigenVectors.Row,k);

            Matrix X = D.T * source;      

            return new PCAEventArgs(X,D);
        }

        public PCAEventArgs SVDFit(Matrix source, int k)
        {
            Matrix matrix = source;
            //1.中心化
            var centralInfo = Matrix.Centralized(matrix);

            Matrix centralizedMatrix = centralInfo.matrix;
            //2.协方差矩阵
            Matrix covMatrix = Matrix.Cov(centralizedMatrix);

            //3.对协方差矩阵求特征值特征向量
            SVDEventArgs result = Math.Linear.Algebra.SVD(covMatrix,k);

            //Matrix eigens = eigResult.Eigen;

            //4.选取有效的特征值
            Matrix eigenVectors = result.V;

            Matrix D = eigenVectors;// Matrix.Clip(eigenVectors, 0, 0, eigenVectors.Row, k);

            Matrix X = D.T * source;

            return new PCAEventArgs(X, D);
        }
    }
}
