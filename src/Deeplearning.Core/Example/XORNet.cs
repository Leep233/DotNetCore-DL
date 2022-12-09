using Deeplearning.Core.Math;
using System;

namespace Deeplearning.Core.Example
{
    public class XORNet
    {
        public Matrix transData = new Matrix(
            
            new Vector(0,0,1,1),
            new Vector(0,1,0,1)
            );
        public Vector realModel = new Vector(0,1,1,0);

        /// <summary>
        /// 模型函数
        /// </summary>
        public Func<Matrix, Matrix, Vector> ModelFunc { get; set; }

        /// <summary>
        /// 损失函数
        /// </summary>
        public Func<Vector, double> LossFunc { get; set; }


        public Func<Matrix, Vector> TargetFunc { get; set; }


        public XORNet()
        {
            ModelFunc = new Func<Matrix,Matrix, Vector>(ModelFunction);

            LossFunc = new Func<Vector, double>(LossFunction);

            TargetFunc = new Func<Matrix, Vector>(TargetFunction);
        }

        private Vector TargetFunction(Matrix arg)
        {
            return realModel;
        }

        private double LossFunction(Vector y)
        {
           int m = y.Length;

            double sum = 0;

           Vector real =  TargetFunction(transData);

            for (int i = 0; i < m; i++)
            {
                sum += MathF.Pow((float)(real[i] - y[i]), 2);
            }


            return sum/m;
        }

        private Vector ModelFunction(Matrix transData, Matrix θ)
        {
            return Matrix.Dot( transData.T , θ.GetVector(0)) + θ.GetVector(1);

        }
    }
}
