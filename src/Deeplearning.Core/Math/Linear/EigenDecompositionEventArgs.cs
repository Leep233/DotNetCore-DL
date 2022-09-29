using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.Linear
{
    public class EigenDecompositionEventArgs:DecomposeEventArgs
    {
        public Vector eigens;

        public Matrix eigenVectors;

        private bool symmetry = false;
        public EigenDecompositionEventArgs(Vector eigen, Matrix vectors,bool isSymmetry = false)
        {
            this.eigens = eigen; 
            this.eigenVectors = vectors;
            symmetry = isSymmetry;
        }

        /// <summary>
        ///对特征值和对应的特征向量进行排序 排序默认从小到大 对比特征值的
        /// </summary>
        /// <param name="order">0：从小到大，其他从大到小</param>
        /// <returns></returns>
        public virtual EigenDecompositionEventArgs Sort(int order = -1)
        {
           var result = MatrixExtension.Sort(eigens,eigenVectors,order);

            this.eigens = result.eigens;

            this.eigenVectors = result.vectors;

            return this;
        }


        /// <summary>
        /// 对多余的特征值进行裁剪，调用此函数时,此函数会自动排序
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public virtual EigenDecompositionEventArgs Clip(int size)
        {
     
            Func<double, double, bool> conditions1 = (a, b) => (MathF.Abs((float)a) < MathF.Abs((float)b));

           var result = MatrixExtension.Sort(eigens, eigenVectors, conditions1);
            //2.裁剪
            Vector values = new Vector(size);
            Vector[] vectors = new Vector[size];

            for (int i = 0; i < size; i++)
            {
                values[i] = result.eigens[i];
                vectors[i] = result.vectors.GetVector(i);//[i];
            }

            Func<double, double, bool> conditions2 = (a, b) =>(a < b);

           var result2 = MatrixExtension.Sort(values, vectors, conditions2);

            eigens = result2.eigens;

            eigenVectors = new Matrix(result2.vectors);

            return this;
        }


        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(" [特征分解] ");
            sb.AppendLine("特征值（Eigen）");
            sb.AppendLine(eigens.ToString());
            sb.AppendLine("特征向量（Vectors）");
            sb.AppendLine(eigenVectors.ToString()); 

            return sb.ToString();
        }

        public override object Validate()
        {
            Matrix D = Matrix.DiagonalMatrix(eigens);
            Matrix inv = symmetry ? eigenVectors.T : Matrix.Inv(eigenVectors);//Vectors.T;//  Vectors.T;// ;
            return eigenVectors * D  * inv;
        }
    }
}
