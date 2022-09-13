using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math.Linear
{
    public class Gradient
    {
        public const float MinValue = 10E-15F;
    
        public static Vector GradientDescent(Func<Vector, float> LinearEquation,Vector initValue,int step = 100, float learningRate = 1E-2f)
        {
            Vector vector = (Vector)initValue.Clone();

            int vectorLength = vector.Length;

            Vector k_Vector = new Vector(vectorLength);

            float z = 0;

            float doubleLR = (2 * learningRate);

            Vector vector1 = new Vector(vectorLength);

            Vector vector2 = new Vector(vectorLength);

            do
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

                double norm = k_Vector.NoSqrtNorm();

                if (norm <= 10E-8) break;

                vector -= (k_Vector * learningRate);

            } while (true);

            return vector;
        }

        public static  async Task<Vector> GradientDescent(Func<double, double> LinearEquation, double initValue =0, Action<GradientInfo> gradientChanged = null,float learningRate = 1E-2f)
        {

            //随机出 开始进行下降的初始点         
            double x = initValue;

            double k = 0;

            double y = 0;

            double learingRate2 = learningRate * 2;

            do
            {
                y = LinearEquation(x + learningRate);

                double y1 = LinearEquation(x - learningRate);

                k = (y - y1) / learingRate2;

                if (gradientChanged != null)
                {
                    GradientInfo info;

                    info.k = (float)k;

                    info.x = (float)x;

                    info.y = (float)y;

                    gradientChanged.Invoke(info);
                }

                await Task.Delay(30);

                //到达可以接受的阈值 跳出函数 说明已经找到了极值
                if (MathF.Abs((float)k) <= 10E-8) break;                

                x -= learningRate * k;

            } while (true);
        

            return new Vector((float)x, (float)y);
        }

       // public static void Jacobian() { }
    }
}
