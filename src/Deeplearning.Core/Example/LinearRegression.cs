using Deeplearning.Core.Math;

namespace Deeplearning.Core.Example
{
    /// <summary>
    /// 线性回归
    /// </summary>
    public class LinearRegression
    {
        /// <summary>
        /// 权重
        /// </summary>
        public Vector weights { get;private set; }
        public Vector bias { get; private set; }
        public LinearRegression()
        {
         
        }

        /// <summary>
        /// 仿射函数
        /// </summary>
        /// <returns></returns>
        public Vector Predice(Matrix input) {

            return Matrix.Dot(input , weights) - bias;
        }
        
        public  double Fit(Matrix data,Vector predict) 
        {
            bias = new Vector(data.Row);

            weights = NormalEquation(data,predict);

            Vector y = Predice(data);

           return InformationTheory.MES(y,predict);
        }

        public Vector NormalEquation(Matrix data, Vector predict) 
        {
            Vector vector = predict + bias;

            Matrix input_T = data.T;

            Matrix matrix = Matrix.Inv(Matrix.Dot(input_T , data));         

            return Matrix.Dot( Matrix.Dot(matrix , input_T) ,vector); 
        }

    }
}
