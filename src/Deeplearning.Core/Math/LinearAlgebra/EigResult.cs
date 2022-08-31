using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math.LinearAlgebra
{
    public class EigResult:DecomposeResult
    {
        public Matrix Eigen { get; set; }

        public Matrix Vectors { get; set; }

        public EigResult()
        {

        }

        public EigResult(Matrix eigen, Matrix vectors)
        {
            this.Eigen = eigen; 
            this.Vectors = vectors;
        }

        /// <summary>
        ///对特征值和对应的特征向量进行排序 排序默认从小到大
        /// </summary>
        /// <param name="order">0：从小到大，其他从大到小</param>
        /// <returns></returns>
        public EigResult Sort(int order = -1)
        {

            int r = (int)MathF.Min(Eigen.Row, Eigen.Column);

            for (int i = 0; i < r - 1; i++)
            {
                for (int j = 0; j < r - 1 - i; j++)
                {
                    int nextIndex = j + 1;

                    bool b = order == -1 ? (Eigen[j, j] < Eigen[nextIndex, nextIndex]) : (Eigen[j, j] > Eigen[nextIndex, nextIndex]);

                    if (b)
                    {
                        float temp = Eigen[j, j];
                        Eigen[j, j] = Eigen[nextIndex, nextIndex];
                        Eigen[nextIndex, nextIndex] = temp;
                        Vector v1 = Vectors.GetVector(nextIndex);
                        Vector v2 = Vectors.GetVector(j);
                        Vectors.Replace(v2, j + 1);
                        Vectors.Replace(v1, j);
                    }
                }
            }
        

            //for (int i = 0; i < r ; i++)
            //{

            //    float value = Eigen[i, i];

            //    Vector v1 = Vectors.GetVector(i);

            //    for (int j = r - i - 1; j >= 0; j--)
            //    {
            //        float f = Eigen[j, j];

            //        if (f >= value) break;

            //        //互换特征值
            //        float temp = value;

            //        Eigen[i, i] = Eigen[j, j];

            //        Eigen[j, j] = temp;

            //        value = f;
            //        //互换特征值对应得特征向量                  

            //        Vector v2 = Vectors.GetVector(j);
            //        Vectors.Replace(v2, j + 1);
            //        Vectors.Replace(v1, j);                
            //    }
            //}


            return this;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.AppendLine("============Eigen Value===========");
            sb.AppendLine(Eigen?.ToString());
            sb.AppendLine("============Eigen Vectors===========");
            sb.AppendLine(Vectors?.ToString());

            return sb.ToString();
        }
    }
}
