using System;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math.Linear
{
    public class Gradient
    {

        public static async Task<Vector> GradientDescentTaskAsync(Func<Vector, float> LinearEquation, Vector initValue, GradientParams @params)
        {
            Vector vector = (Vector)initValue;

            int vectorLength = vector.Length;

            Vector k_Vector = new Vector(vectorLength);

            float learningRate = @params.rate;

            float doubleLR = (2 * learningRate);

            Vector vector1 = new Vector(vectorLength);

            Vector vector2 = new Vector(vectorLength);

           await Task.Factory.StartNew(async () => {
                for (int i = 0; i < @params.step; i++)
                {
                    for (int j = 0; j < vectorLength; j++)
                    {
                        vector1[j] = vector[j] - learningRate;
                        vector2[j] = vector[j] + learningRate;

                        for (int k = 0; k < vectorLength; k++)
                        {
                            if (j == k) continue;
                            vector1[k] = vector[k];
                            vector2[k] = vector[k];
                        }
                        k_Vector[j] = (LinearEquation(vector2) - LinearEquation(vector1)) / doubleLR;
                    }

                    double norm = Vector.NoSqrtNorm(k_Vector,2);

                    if (norm <= @params.e) break;

                    vector -= (k_Vector * learningRate);
                   await Task.Delay(1000);
                }

            });


            return vector;
        }

        public static Vector GradientDescent(Func<Vector, float> LinearEquation, Vector initValue, GradientParams @params)
        {
            Vector vector = initValue;

            int vectorLength = vector.Length;

            Vector k_Vector = new Vector(vectorLength);

            float learningRate = @params.rate;

            float doubleLR = (2 * learningRate);

            Vector vector1 = new Vector(vectorLength);

            Vector vector2 = new Vector(vectorLength);

            for (int i = 0; i < @params.step; i++)
            {
                for (int j = 0; j < vectorLength; j++)
                {
                    vector1[j] = vector[j] - learningRate;
                    vector2[j] = vector[j] + learningRate;

                    for (int k = 0; k < vectorLength; k++)
                    {
                        if (j == k) continue;
                        vector1[k] = vector[k];
                        vector2[k] = vector[k];
                    }
                    k_Vector[j] = (LinearEquation(vector2) - LinearEquation(vector1)) / doubleLR;
                }

                double norm = Vector.NoSqrtNorm(k_Vector,2);

                if (norm <= @params.e) break;

                vector -= (k_Vector * learningRate);
            }


            return vector;
        }

        public static async Task<Vector> GradientDescent(Func<double, double> LinearEquation, double initValue, GradientParams @params, Action<GradientEventArgs> gradientChanged = null)
        {

            //随机出 开始进行下降的初始点         
            double x = initValue;

            double k = 0;

            double y = 0;

            float learningRate = @params.rate;

            double learingRate2 = learningRate * 2;

            for (int i = 0; i < @params.step; i++)
            {
                y = LinearEquation(x + learningRate);

                double y1 = LinearEquation(x - learningRate);

                k = (y - y1) / learingRate2;

                if (gradientChanged != null)
                {
                    GradientEventArgs eventArgs;

                    eventArgs.k = (float)k;

                    eventArgs.x = (float)x;

                    eventArgs.y = (float)y;

                    gradientChanged.Invoke(eventArgs);
                }

                await Task.Delay(30);

                //到达可以接受的阈值 跳出函数 说明已经找到了极值
                if (MathF.Abs((float)k) <= @params.e) break;

                x -= learningRate * k;
            }

            return new Vector((float)x, (float)y);
        }

        public static float SGD(Vector[] inputs, GradientParams @params)
        {
            //样本数
            int m = 0;

            //线性组合函数
            Func<Vector, Matrix, Vector> linearFunction = (x,w) => {
                Vector v =  w.T * x ;

              return v;
            };
            //权重
            Vector w = new Vector(inputs.Length);

          //  linearFunction(inputs,);

            return 0;

        }
    }

    public struct GradientParams
    {
        /// <summary>
        /// 计算步数
        /// </summary>
        public int step;
        /// <summary>
        /// 学习率
        /// </summary>
        public float rate;
        /// <summary>
        /// 最小
        /// </summary>
        public float e;

        //public GradientParams()
        //{
        //    this.step = 100;
        //    this.rate = 0.01f;
        //    this.e = 10E-8F;
        //}

        public GradientParams(int step, float lr, float e)
        {
            this.step = step;
            this.rate = lr;
            this.e = e;
        }

        public static GradientParams Default => new GradientParams(1000, 0.005f, 10E-8F);

    }

}
