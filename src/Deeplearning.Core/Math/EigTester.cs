using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Deeplearning.Core.Math
{
    /// <summary>
    /// 特征分解测试 
    /// </summary>
    public class EigTest
    {
        public static async void Eig(Matrix squareMatrix) 
        {
            if (!squareMatrix.IsSquare()) return;

            int size = squareMatrix.Rows;

            int step = 100;

            Vector v = new Vector(-1,5,3);

            Matrix temp = squareMatrix.Clone() as Matrix;

            StringBuilder stringBuilder = new StringBuilder();

            int []  r = new int[] {1,3,5,10 };

            for (int i = 0; i < r.Length; i++)
            {
                for (int j = r[i]; 0 < j; j--)
                {
                    temp = temp * j;

                    float[] b = temp * v;

                    float sum = 0;

                    for (int k   = 0; k < b.Length; k++)
                    {
                        sum += b[k];
                    }

                    for (int k = 0; k < b.Length; k++)
                    {
                        stringBuilder.Append((b[k] / sum).ToString("F8"));
                    }

                    Debug.WriteLine(stringBuilder.ToString());

                    await System.Threading.Tasks.Task.Delay(500);

                    stringBuilder.Clear();
                }
            }
        }
    }
}
