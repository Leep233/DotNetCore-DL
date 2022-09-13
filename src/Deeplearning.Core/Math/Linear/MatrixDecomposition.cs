using Deeplearning.Core.Attributes;
using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public class MatrixDecomposition
    {

        public static SVDResult SVD(Matrix source, Func<Matrix, QRResult> decompose, int step = 15)
        {   
            EigResult A_Eig = Eig(source.T * source, decompose, step);
    
            Matrix V = A_Eig.Vectors;

            EigResult B_Eig = Eig(source * source.T, decompose, step);
 
            Matrix U = B_Eig.Vectors;

            float[] eigens = A_Eig.Eigen.DiagonalElements();

            int r = U.Column;

            int c = V.Column;

            Matrix D = new Matrix(r, c);

            int l = (int)MathF.Min(r, c);

            l = (int)MathF.Min(l,eigens.Length);

            for (int i = 0; i < l; i++)
            {
                float value = Validator.ZeroValidation(eigens[i]);
                 
                D[i, i] = value==0?0: MathF.Sqrt((float)value);
            }
            return new SVDResult(U,D,V);
        }


        /// <summary>
        /// 主成分分析（Principal Component Analysis，PCA）
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>   
        public static PCAResult PCA(Matrix source,int r, Func<Matrix, QRResult> qrDecFunction)
        {
    
            //1.中心化
            var centralInfo= source.Centralized();
            Matrix centralizedMatrix = centralInfo.matrix;
            //2.协方差矩阵
            Matrix covMatrix = source.Cov();
      
            //3.对协方差矩阵求特征值特征向量
            EigResult result = Eig(covMatrix, qrDecFunction);
           
            // Matrix eigens = eigResult.Eigen;

            //4.选取有效的特征值

            Matrix eigenVectors = result.Vectors;// result.V;//eigResult.Vectors;
           
            Matrix P = eigenVectors.Clip(0,0, r, eigenVectors.Column);

       
            Matrix y = P * source ;
   
            return new PCAResult();
        }


        public static EigResult Eig(Matrix source, Func<Matrix,QRResult> qrDecFunction, int step = 15)
        {

            Matrix matrix = (Matrix)source.Clone();

            int k = (int)MathF.Min(matrix.Row, matrix.Column);

            Matrix Q = Matrix.UnitMatrix(k);

            for (int i = 0; i < step; i++)
            {
                QRResult result = qrDecFunction(matrix);

                matrix = result.R * result.Q;

                Q = Q * result.Q;
            }

            for (int i = 0; i < matrix.Row; i++)
            {
                for (int j = 0; j < matrix.Column; j++)
                {
                    if (i == j) continue;
                    matrix[i, j] = 0;
                }
            }

            return new EigResult(matrix, Q).Sort();
        }

    }
}
