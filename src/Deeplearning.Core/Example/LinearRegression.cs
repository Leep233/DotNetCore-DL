using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Core.Math.Models;
using Deeplearning.Core.Math.Probability;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

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

            return input * weights - bias;
        }
        
        public  double Train(Matrix data,Vector predict) 
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

            Matrix matrix = (input_T * data).inverse;         

            return matrix * input_T * vector; 
        }

      

    }
}
