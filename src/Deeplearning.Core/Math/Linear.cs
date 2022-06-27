using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math
{
    public static class Linear
    {
        public const float MinValue = 0.0001f;
        public static Task GradientDescentTaskAsync(int step, Func<Vector2D, float> original, Action<Gradient3DInfo> gradientChanged)
        {



            return Task.Run(() =>
            {


                Random random = new Random();

                Vector2D vector2;

                vector2.x = random.Next(-10, 10);

                vector2.y = random.Next(-10, 10);

                float learningRate = 0.05f;

                float minValue = 0.0001f;

                Vector2D k_Vector;

                float z = 0;

                k_Vector.x = 1;

                k_Vector.y = 1;

                float doubleLR = (2 * learningRate);

                for (int i = 0; i < step; i++)
                {
                    Vector2D tempV2;

                    Vector2D tempV1;

                    z = original(vector2);

                    tempV2 = new Vector2D()
                    {
                        x = vector2.x + learningRate,
                        y = vector2.y
                    };

                    tempV1 = new Vector2D()
                    {
                        x = vector2.x - learningRate,
                        y = vector2.y
                    };

                    float k_x = (original(tempV2) - original(tempV1)) / doubleLR;

                    tempV2 = new Vector2D()
                    {
                        x = vector2.x,
                        y = vector2.y + learningRate
                    };

                    tempV1 = new Vector2D()
                    {
                        x = vector2.x,
                        y = vector2.y - learningRate
                    };

                    float k_y = (original(tempV2) - original(tempV1)) / doubleLR;

                    k_Vector.x = k_x;

                    k_Vector.y = k_y;

                    if (gradientChanged != null)
                    {
                        Gradient3DInfo gradient3DInfo = new Gradient3DInfo();

                        gradient3DInfo.x = vector2.x;

                        gradient3DInfo.y = vector2.y;

                        gradient3DInfo.z = z;

                        gradient3DInfo.grad = k_Vector;

                        gradientChanged.Invoke(gradient3DInfo);

                    }

                    if (MathF.Abs(k_Vector.x) <= minValue && MathF.Abs(k_Vector.y) <= minValue)
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
