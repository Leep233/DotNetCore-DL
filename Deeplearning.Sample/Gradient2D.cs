using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Sample
{
    public class Gradient2D
    {
   
        /// <summary>
        /// 学习率
        /// </summary>
        public float LearningRate { get; set; } = 0.01f;
        /// <summary>
        /// 最低阈值
        /// </summary>
        public float ThresholdValue { get; set; } = 0.0001f;

        public event EventHandler<GradientInfo> GradientChangedEvent;

        /// <summary>
        /// 原函数
        /// </summary>
        private Func<double, double> original;

        private Random random = new Random();

        public Gradient2D(Func<double, double> original)
        {
            this.original = original;
        }

        /// <summary>
        /// 梯度下降
        /// </summary>
        /// <param name="step">下降次数</param>
        /// <param name="learningRate">学习率</param>
        /// <param name="thresholdValue">极值阈值</param>
        /// <param name="gradientChangedEvent"></param>
        /// <returns></returns>
        public async Task GradientDescentTaskAsync(int step, double learningRate=0.01, double thresholdValue = 0.0001) {

            //随机出 开始进行下降的初始点         
            double x =  random.Next(-10, 10);

            double y;

            double k = 0;

            for (int i = 0; i < step; i++)
            {
                 y = original(x + learningRate);

                double y1 = original(x - learningRate);

                k = ((y - y1) / (learningRate * 2));

                if (GradientChangedEvent != null)
                {
                    GradientInfo info;

                    info.k = k;

                    info.x = x;

                    info.y = y;

                    GradientChangedEvent.Invoke(this, info);
                }

                //到达可以接受的阈值 跳出函数 说明已经找到了极值
                if (Math.Abs(k) <= thresholdValue)
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
        public async Task GradientDescentTaskAsync(int step, Func<double, double> derivative, double learningRate = 0.01,double thresholdValue = 0.0001) 
        {
            //随机出 开始进行下降的初始点         
           double x = random.Next(-10, 10);

           double y = original(x); 

            double k = 0;

            for (int i = 0; i < step; i++)
            {
                y = original(x);

                //求导/斜率
                k = derivative(x);

                if(GradientChangedEvent != null)
                {
                    GradientInfo info;

                    info.k = k;

                    info.x = x;

                    info.y = y;

                    GradientChangedEvent.Invoke(this, info);
                }

                //到达可以接受的阈值 跳出函数 说明已经找到了极值
                if (Math.Abs(k) <=thresholdValue) 
                {
                    break;
                }

                await Task.Delay(33);

                x -=  (learningRate * k);
            }
        }

        public struct GradientInfo
        {
            // x 
            public double x;
            // y
            public double y;
            // 斜率
            public double k;

            public override string ToString()
            {
                return $"({x.ToString("F4")},{y.ToString("F4")}) 斜率/导数：{k.ToString("F4")}";
            }
        }
    }
    

}
