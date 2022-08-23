using Deeplearning.Core.Math.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

namespace Deeplearning.Core.Math
{
    public static class LinearAlgebra
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

        public static (Matrix Q, Matrix R) HouseholderTest(Matrix source)
        {
            Matrix matrix = (Matrix)source.Clone();

            StringBuilder logBuilder = new StringBuilder();

            logBuilder.AppendLine($"===============[Source Matrix]===============");
            logBuilder.AppendLine($"{matrix}");

            int k = (int)MathF.Min(matrix.Rows, matrix.Columns);

            logBuilder.AppendLine($"k = {k}");

            Matrix E = new Matrix(matrix.Rows, matrix.Columns);

            for (int i = 0; i < k; i++)
            {
                E[i, i] = 1;
            }

            for (int i = 0; i < k; i++)
            {
                Vector x = matrix.GetVector(0);
                Vector y = new Vector(x.Length);
                y[0] = x.Norm(2);
                Vector z = x - y;
                Vector w = z / z.Norm(2);

                logBuilder.AppendLine($"===============[{i}]===============");
                logBuilder.AppendLine($"x = {x},y = {y},w = {w}");


                logBuilder.AppendLine($"===============[E Matrix]===============");
                logBuilder.AppendLine($"{E}");

                Matrix unit = Matrix.UnitMatrix(x.Length);

                Matrix h = unit - (2 * w * w.T);
                logBuilder.AppendLine($"===============[H Matrix]===============");
                logBuilder.AppendLine($"{h}");
                Matrix r = h * matrix;
                logBuilder.AppendLine($"===============[R Matrix]===============");
                logBuilder.AppendLine($"{r}");

                matrix = matrix.AlgebraicCofactor(i, i);

                for (int m = 0; m < matrix.Rows; m++)
                {
                    for (int n = 0; n < matrix.Columns; n++)
                    {
                        E[m + i + 1, n + i + 1] = matrix[m, n];
                    }
                }

                logBuilder.AppendLine($"=================================");
            }


            return (null, null);// (Q,R);    

        }
        public static (Matrix Q, Matrix R) Householder(Matrix source) 
        {
          
 

            Vector[] vectors = new Vector[4];
            vectors[0] = new Vector(1,1,1,1);
            vectors[1] = new Vector(2,1,1,1);
            vectors[2] = new Vector(3,2,1,1);
            vectors[3] = new Vector(4,3,2,1);

            source = new Matrix(vectors);

            Matrix matrix = (Matrix)source.Clone();       

            List<Matrix> qs = new List<Matrix>();

            for (int i = -1; i < matrix.Columns; i++)
            {
                matrix = matrix.AlgebraicCofactor(i, i);

                // if(matrix)

                Vector x = matrix.GetVector(0);

                double x_norm = x.Norm(2);

                Vector y = new Vector(x.Length);

                y[0] = x_norm;

                Vector v = x - y;

                Vector w = v / v.Norm(2);

                Matrix I = Matrix.UnitMatrix(x.Length);

                Matrix _h = I - (2 * w * w.T);

                _h *= matrix;


                if (qs.Count <= 0)
                {
                    qs.Add(_h);
                }
                else
                {
                    Matrix q = (Matrix)qs[qs.Count - 1].Clone();

                    int index = i + 1;

                    for (int r = 0; r < _h.Rows; r++)
                    {
                        for (int c = 0; c < _h.Columns; c++)
                        {
                            q[r + index, c + index] = _h[r, c];
                        }
                    }
                    qs.Add(q);
                }
            }

            Matrix Q = qs[0];


            for (int i = 1; i < qs.Count; i++)
            {
                Q *= qs[i];
            }

            return  (Q,R: Q * source);
          
        }

       

        public static void Givens(Matrix source) { }
        /// <summary>
        /// 传统格拉姆 斯密特正交
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static (Matrix Q,Matrix R) CGS(Matrix source) {

            Vector[] srcVector = source.Vectors();

            Vector[] bs = new Vector[srcVector.Length];

            int vectorSize = srcVector[0].Length;
            int vectorCount = srcVector.Length;

            for (int i = 0; i < vectorCount; i++) 
            { 
                Vector target = srcVector[i];

                Vector temp = new Vector(vectorSize);

                for (int j = i - 1; j >= 0; j--)
                {
                    Vector v = bs[j];          
                    double value =  (target * v) / (v * v);
                    temp += value * v;
                }              
                bs[i] = target - temp;
                bs[i] = bs[i]/ bs[i].Norm(2);
            }

            Matrix Q = new Matrix(bs);

            Matrix R = Q.T * source;

            return (Q, R);
        }


        public static (Matrix egin, Matrix vectors) Eig(Matrix source,Func<Matrix,(Matrix Q,Matrix R)> qrDecFunction,int step = 100) {

            Matrix matrix = (Matrix)source.Clone();

            int k = (int)MathF.Min(matrix.Rows, matrix.Columns);

            Matrix Q = Matrix.UnitMatrix(k);

            Matrix ak = null;

            for (int i = 0; i < step; i++)
            {
                var qr = qrDecFunction.Invoke(matrix);

                matrix = qr.R * qr.Q;

                Q = Q * qr.Q;

                ak = qr.Q * qr.R;
            }

            int size = ak.Rows;            

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    if (i != j)
                    {
                        ak[i, j] = 0;
                    }
                    else {

                        for (int m = i - 1; m >= 0; m--)
                        {
                            if (ak[m, m] < ak[i, i])
                            {
                                double temp = ak[i, i];
                                ak[i, i] = ak[m, m];
                                ak[m, m] = temp;

                                for (int r = 0; r < Q.Rows; r++)
                                {
                                    temp = Q[r, i];
                                    Q[r, i] = Q[r, m];
                                    Q[r, m] = temp;
                                }
                            }
                        }

                    }

                }
            }
            return (egin: ak, vectors: Q);    
        }

        /// <summary>
        /// 改良Modified Gram-Schmidt
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static (Matrix Q, Matrix R) MGS(Matrix source) {

            Vector[] b_Vectors = source.Vectors();

            int vectorSize = b_Vectors[0].Length;

            int vectorCount = b_Vectors.Length;

            Vector[] e_Vectors =new  Vector[vectorCount];
      
            e_Vectors[0] = b_Vectors[0] / b_Vectors[0].Norm(2);

            for (int i = 0; i < vectorCount; i++)
            {
                Vector b = b_Vectors[i];

                Vector e = b / b.Norm(2);

                for (int j = i + 1; j < vectorCount; j++)
                {
                    b = b_Vectors[j];
                    b_Vectors[j] = b - (b * e * e);
                }
                e_Vectors[i] = e;
            }

            Matrix Q = new Matrix(e_Vectors);

            Matrix R = Q.T * source;

            return (Q,R);
        }


      
    }
}
