using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math.LinearAlgebra
{
    public class Gradient
    {
        public const float MinValue = 0.0001f;
        public static Task GradientDescentTaskAsync(int step, Func<Vector, float> original, Action<Gradient3DInfo> gradientChanged)
        {
            return Task.Run(() =>
            {
                Vector vector2 = Vector.Random(2, -10, 10);

                float learningRate = 0.05f;

                float minValue = 0.0001f;

                Vector k_Vector = Vector.One(2);

                float z = 0;

                float doubleLR = (2 * learningRate);

                for (int i = 0; i < step; i++)
                {
                    Vector tempV2;

                    Vector tempV1;

                    z = original(vector2);

                    tempV2 = new Vector(vector2[0] + learningRate, vector2[1]);


                    tempV1 = new Vector(vector2[0] - learningRate, vector2[1]);


                    float k_x = (original(tempV2) - original(tempV1)) / doubleLR;

                    tempV2 = new Vector(vector2[0], vector2[1] + learningRate);


                    tempV1 = new Vector(vector2[0], vector2[1] - learningRate);


                    float k_y = (original(tempV2) - original(tempV1)) / doubleLR;

                    k_Vector[0] = k_x;

                    k_Vector[1] = k_y;

                    if (gradientChanged != null)
                    {
                        Gradient3DInfo gradient3DInfo = new Gradient3DInfo();

                        gradient3DInfo.x = (float)vector2[0];

                        gradient3DInfo.y = (float)vector2[1];

                        gradient3DInfo.z = z;

                        gradient3DInfo.grad = k_Vector;

                        gradientChanged.Invoke(gradient3DInfo);

                    }

                    if (MathF.Abs((float)vector2[0]) <= minValue && MathF.Abs((float)vector2[1]) <= minValue)
                    {
                        break;
                    }

                    vector2 -= (k_Vector * learningRate);
                }
            });
        }

        public static async Task GradientDescentTaskAsync(float initX, int step, Func<double, double> original, Action<GradientInfo> gradientChanged, float learningRate = 0.01f)
        {

            //随机出 开始进行下降的初始点         
            float x = initX;

            float y;

            float k = 0;

            for (int i = 0; i < step; i++)
            {
                y = (float)original(x + learningRate);

                float y1 = (float)original(x - learningRate);

                k = ((y - y1) / (learningRate * 2));

                if (gradientChanged != null)
                {
                    GradientInfo info;

                    info.k = k;

                    info.x = x;

                    info.y = y;

                    gradientChanged.Invoke(info);
                }

                //到达可以接受的阈值 跳出函数 说明已经找到了极值
                if (MathF.Abs(k) <= MinValue)
                {
                    break;
                }

                await Task.Delay(33);

                x -= learningRate * k;
            }
        }

        /// <summary>
        /// 进行梯度下降
        /// </summary>
        /// <param name="desCount">计算下降的次数</param>
        /// <param name="gradientChangedEvent">每次下降发生的变化事件</param>
        /// <param name="descRate">下降率</param>
        /// <param name="thresholdValue">阈值</param>
        public static async Task GradientDescentTaskAsync(float initX, int step, Func<double, double> original, Func<double, double> derivative, Action<GradientInfo> gradientChanged, float learningRate = 0.01f)
        {
            //随机出 开始进行下降的初始点         
            float x = initX;

            float y = (float)original(x);

            float k = 0;

            for (int i = 0; i < step; i++)
            {
                y = (float)original(x);

                //求导/斜率
                k = (float)derivative(x);

                if (gradientChanged != null)
                {
                    GradientInfo info;

                    info.k = k;

                    info.x = x;

                    info.y = y;

                    gradientChanged.Invoke(info);
                }

                //到达可以接受的阈值 跳出函数 说明已经找到了极值
                if (MathF.Abs(k) <= MinValue)
                {
                    break;
                }

                await Task.Delay(33);

                x -= (learningRate * k);
            }
        }
    }
}
