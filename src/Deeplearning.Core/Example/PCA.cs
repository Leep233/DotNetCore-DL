using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Core.Math.Probability;

namespace Deeplearning.Core.Example
{
    public class PCA
    {

        public PCAEventArgs EigFit(Matrix source,int k) 
        {
          

            Matrix matrix = Matrix.Copy(source);

            //1.中心化
             var centralInfo = ProbabilityDistribution.MinMaxScaler(matrix);

            //var centralInfo = Matrix.Centralized(matrix);

            Matrix centralizedMatrix = centralInfo;// centralInfo.matrix;// source;//
            //2.协方差矩阵
            Matrix covMatrix = ProbabilityDistribution.Cov(centralizedMatrix);

            //3.对协方差矩阵求特征值特征向量
            EigenDecompositionEventArgs result =Algebra.Eig(covMatrix,k);

            //4.选取有效的特征值
            Matrix D = result.eigenVectors;

            Matrix X = Matrix.Dot(source, D);      

            return new PCAEventArgs(X,D);
        }

        public PCAEventArgs SVDFit(Matrix source, int k)
        {
            Matrix matrix = Matrix.Copy(source);

            //1.中心化
            var centralInfo = ProbabilityDistribution.MinMaxScaler(matrix);

            Matrix centralizedMatrix = centralInfo;//  centralInfo.matrix; //matrix;//
            //3.SVD
            SVDEventArgs result =Algebra.SVD(centralizedMatrix, k);

            //4.选取有效的特征值 
            Matrix D = result.V;

            Matrix X = Matrix.Dot(source,D);

            return new PCAEventArgs(X, D);
        }
    }
}
